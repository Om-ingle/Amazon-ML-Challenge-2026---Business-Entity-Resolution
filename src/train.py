"""Phase 8/10/11 — train the pair matcher and calibrate the entity-level rule.

Trains on pairs from the FIT split, evaluates the full pipeline (blocking ->
model -> entity decision) on the VAL split with macro F0.5, and sweeps the
decision threshold.
"""
from __future__ import annotations

import json
import pickle
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import features as F  # noqa: E402
from eval_blocking import make_split, load_gt  # noqa: E402
from pairs import build_pairs, BLOCK  # noqa: E402

EXP = os.path.join(C.CACHE, "..", "experiments")


def entity_predict(pr: dict, keep_fn):
    """Turn per-pair probabilities into {s1: set(tgt)} using keep_fn."""
    out = {}
    for s, t, p in zip(pr["s1_id"], pr["tgt_id"], pr["prob"]):
        if keep_fn(p, s):
            out.setdefault(s, set()).add(t)
    return out


def scores_for(pr, keys):
    """{s1: (best_prob, n_cand)} diagnostics."""
    d = {}
    for s, p in zip(pr["s1_id"], pr["prob"]):
        cur = d.get(s)
        if cur is None or p > cur:
            d[s] = float(p)
    return d


def sweep(pr, gt, keys, thresholds, bij=False):
    """Entity-level decision sweep.

    bij=False: keep every pair with p >= threshold.
    bij=True : first resolve the S2/S3 side as an assignment (each target record may
               back at most one Source-1 entity -- holds in the training ground
               truth, where every matched id appears exactly once), keeping only the
               highest-probability claimant, then apply the threshold.
    """
    rows = []
    if bij:
        # best claimant per target id
        order = np.lexsort((-pr["prob"], pr["tgt_id"]))
        s_sorted = pr["tgt_id"][order]
        first = np.ones(len(order), bool)
        first[1:] = s_sorted[1:] != s_sorted[:-1]
        winner = order[first]
        p2 = dict(s1_id=pr["s1_id"][winner], tgt_id=pr["tgt_id"][winner],
                  prob=pr["prob"][winner])
    else:
        p2 = pr
    for th in thresholds:
        pred = entity_predict(p2, lambda p, s, th=th: p >= th)
        m = C.macro_f05(pred, gt, keys)
        m["threshold"] = th
        m["avg_pred_per_s1"] = round(
            sum(len(v) for v in pred.values()) / max(len(keys), 1), 3)
        rows.append(m)
    return rows


def main():
    t_all = time.time()
    val, fit = make_split()
    n_fit = int(os.environ.get("FIT_N", "40000"))
    n_val = int(os.environ.get("VAL_N", "20000"))
    fit_ids, val_ids = fit[:n_fit], val[:n_val]
    gt_fit = load_gt(fit_ids)
    gt_val = load_gt(val_ids)
    print(f"fit={len(fit_ids)} val={len(val_ids)}"
          f" (fit pairs={sum(len(v) for v in gt_fit.values())},"
          f" val pairs={sum(len(v) for v in gt_val.values())})", flush=True)
    print(f"blocking={BLOCK}", flush=True)

    # No process pool for the string metrics: they cost ~1.7e-5 s/pair, so a few
    # hundred core-seconds for the whole fit+val build, while shipping the string
    # columns to workers dominated both memory and wall-clock (and a worker killed
    # by memory pressure makes multiprocessing.Pool hang instead of raising).
    t0 = time.time()
    tr = build_pairs("train", fit_ids, gt=gt_fit, pool=None)
    print(f"train pairs built in {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    va = build_pairs("train", val_ids, gt=gt_val, pool=None)
    print(f"val pairs built in {time.time()-t0:.0f}s", flush=True)

    y = tr["label"]
    X = tr["feats"]
    print(f"train: {X.shape} positives={y.sum():,} ({100*y.mean():.2f}%)", flush=True)

    # class-balanced negative downsampling keeps training cheap without losing signal
    rng = np.random.default_rng(0)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_neg = min(len(neg), max(3 * len(pos), 500_000))
    sel = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(sel)
    Xt, yt = X[sel], y[sel]
    print(f"train subsample: {Xt.shape} pos={yt.sum():,}", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    sc = StandardScaler().fit(Xt)
    lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1)
    lr.fit(sc.transform(Xt), yt)

    spw = (yt == 0).sum() / max((yt == 1).sum(), 1)
    n_est = int(os.environ.get("N_EST", "800"))
    xg = xgb.XGBClassifier(
        n_estimators=n_est, max_depth=int(os.environ.get("MAX_DEPTH", "8")),
        learning_rate=0.07,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_lambda=2.0, scale_pos_weight=float(np.sqrt(spw)),
        tree_method="hist", n_jobs=14, eval_metric="logloss",
        early_stopping_rounds=50,
    )
    cut = int(0.9 * len(Xt))
    xg.fit(Xt[:cut], yt[:cut], eval_set=[(Xt[cut:], yt[cut:])], verbose=False)
    print(f"xgb best_iteration={xg.best_iteration}", flush=True)

    imp = sorted(zip(F.FEATURE_NAMES, xg.feature_importances_),
                 key=lambda z: -z[1])
    print("top features:", ", ".join(f"{k}={v:.3f}" for k, v in imp[:15]), flush=True)

    Xv = va["feats"]
    probs = {}
    probs["lr"] = lr.predict_proba(sc.transform(Xv))[:, 1].astype(np.float32)
    probs["xgb"] = xg.predict_proba(Xv)[:, 1].astype(np.float32)
    probs["blend"] = (0.35 * probs["lr"] + 0.65 * probs["xgb"]).astype(np.float32)

    keys = list(val_ids)
    out = {}
    ths = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97,
           0.99, 0.995]
    for name, pv in probs.items():
        pr = dict(va, prob=pv)
        for bij in (False, True):
            rows = sweep(pr, gt_val, keys, ths, bij=bij)
            out[f"{name}|bij={int(bij)}"] = rows
            best = max(rows, key=lambda r: r["f05_macro"])
            print(f"\n[{name} bij={int(bij)}] best th={best['threshold']} "
                  f"F05={best['f05_macro']:.4f} P={best['micro_precision']:.4f} "
                  f"R={best['micro_recall']:.4f} singR={best['singleton_recall']:.3f} "
                  f"singP={best['singleton_precision']:.3f}")
            for r in rows:
                print(f"   th={r['threshold']:.3f} F05={r['f05_macro']:.4f} "
                      f"P={r['micro_precision']:.4f} R={r['micro_recall']:.4f} "
                      f"avgPred={r['avg_pred_per_s1']:.2f}")

    os.makedirs(EXP, exist_ok=True)
    with open(os.path.join(EXP, "train_sweep.json"), "w") as f:
        json.dump(out, f, indent=1)
    # cache val artifacts so error analysis costs nothing to re-run
    np.savez(os.path.join(C.CACHE, "val_pairs.npz"),
             s1_id=va["s1_id"], tgt_id=va["tgt_id"], tgt_src=va["tgt_src"],
             label=va["label"], feats=va["feats"],
             prob_xgb=probs["xgb"], prob_lr=probs["lr"],
             feature_names=np.array(F.FEATURE_NAMES))

    # persist the best configuration for predict.py
    best_name, best_key, best_row = None, None, None
    for k, rows in out.items():
        for r in rows:
            if best_row is None or r["f05_macro"] > best_row["f05_macro"]:
                best_row, best_key = r, k
    best_name = best_key.split("|")[0]
    best_bij = best_key.endswith("bij=1")
    cfg = dict(
        model=best_name, blocking=BLOCK, threshold=best_row["threshold"],
        blend_lr={"lr": 1.0, "xgb": 0.0, "blend": 0.35}.get(best_name, 0.0),
        val_f05=best_row["f05_macro"], val_precision=best_row["micro_precision"],
        val_recall=best_row["micro_recall"], n_fit=len(fit_ids),
        n_val=len(val_ids), feature_version=F.FEATURE_VERSION,
        feature_names=F.FEATURE_NAMES, bij=best_bij,
    )
    bd = os.path.join(EXP, "best")
    os.makedirs(bd, exist_ok=True)
    with open(os.path.join(bd, "model.pkl"), "wb") as f:
        pickle.dump(dict(xgb=xg, lr=lr, scaler=sc), f)
    with open(os.path.join(bd, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"\nBEST: {best_name} th={cfg['threshold']} F05={cfg['val_f05']:.4f} "
          f"P={cfg['val_precision']:.4f} R={cfg['val_recall']:.4f}")
    print(f"saved {bd}/model.pkl, {bd}/config.json")
    print(f"total {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
