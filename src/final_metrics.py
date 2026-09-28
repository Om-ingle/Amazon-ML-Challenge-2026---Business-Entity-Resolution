"""Phase 18/19 -- final metrics from cached artifacts, fully vectorised.

Reads cache/val_pairs.npz (candidate pairs with features, labels and model
probabilities) plus the index metadata, so nothing is recomputed. Grouping is
done with integer codes rather than object-dtype sorts: sorting millions of
Python strings is orders of magnitude slower than sorting their factorized codes.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
from eval_blocking import load_gt, make_split  # noqa: E402

val, fit = make_split()
val_ids = list(val[:30_000])
gt = load_gt(val_ids)
z = np.load(os.path.join(C.CACHE, "val_pairs.npz"), allow_pickle=True)
cfg = json.load(open(os.path.join(C.CACHE, "..", "experiments", "best",
                                  "config.json")))
TH = cfg["threshold"]

df = pd.DataFrame({"s1": z["s1_id"], "tg": z["tgt_id"],
                   "p": z["prob_xgb"].astype(np.float64),
                   "y": z["label"].astype(np.int32)})
# `scode` must be the position of the S1 in val_ids, not an independent
# factorization: the per-entity arrays are aligned by that index, so any other
# encoding silently misaligns prediction counts with truth counts (micro
# precision/recall still look right, because they are order-independent sums).
pos = {e: i for i, e in enumerate(val_ids)}
df["scode"] = df["s1"].map(pos).to_numpy()
df["tcode"] = pd.factorize(df["tg"])[0]
ntrue = pd.Series({e: len(gt.get(e, ())) for e in val_ids})
print(f"val S1={len(val_ids):,}  candidate pairs={len(df):,}  th={TH}")

# ---- candidate-set (blocking) metrics over the pairs the model actually saw
nc = df.groupby("scode").size().reindex(np.arange(len(val_ids))).fillna(0).to_numpy()
hit = df.join(ntrue.rename("nt"), on="s1")
tp_c = int(hit.loc[hit.y == 1].groupby("s1").size().reindex(val_ids).fillna(0).sum())
tot_c = int(ntrue.sum())
per = (hit[hit.y == 1].groupby("s1").size().reindex(val_ids).fillna(0)
       / ntrue.replace(0, np.nan)).dropna()
pool = {}
for f in glob.glob(os.path.join(C.CACHE, "indexes", "train_s*_i2.npz")):
    _, s, ctry, _ = os.path.basename(f).replace(".npz", "").split("_")
    pool[(s, ctry)] = int(np.load(f)["n"][0])
tot_pool = sum(pool.values())
print(f"\ncandidate recall (micro) = {tp_c/tot_c:.4f}")
print(f"candidate recall (macro) = {per.mean():.4f}")
print(f"avg candidates/S1        = {nc.mean():.1f}")
print(f"median / p95 / max       = {int(np.median(nc))} / "
      f"{int(np.percentile(nc,95))} / {int(nc.max())}")
print(f"zero-candidate S1        = {100*(nc==0).mean():.2f}%")
print(f"search pool              = {tot_pool:,} S2+S3 records, "
      f"{len({c for _,c in pool})} countries")
print(f"reduction ratio          = {1-nc.mean()/(tot_pool/len({c for _,c in pool})):.8f}")


def decide(th, bij):
    """Per-S1 (npred, tp) after optional assignment, as aligned arrays."""
    pr = df[df.p >= th]
    if bij:
        pr = (pr.sort_values(["tcode", "p"], ascending=[True, False])
                .drop_duplicates("tcode"))
    g = pr.groupby("scode").agg(npred=("y", "size"), tp=("y", "sum"))
    idx = np.arange(len(val_ids))
    return (g["npred"].reindex(idx).fillna(0).to_numpy(),
            g["tp"].reindex(idx).fillna(0).to_numpy())


def f05_macro(npred, tp, ntrue_a):
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(npred > 0, tp / np.maximum(npred, 1), 0.0)
        r = np.where(ntrue_a > 0, tp / np.maximum(ntrue_a, 1), 0.0)
        f = np.where((p + r) > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    empty_both = (ntrue_a == 0) & (npred == 0)
    f = np.where(empty_both, 1.0, f)
    return f.mean(), tp.sum() / max(npred.sum(), 1), tp.sum() / max(ntrue_a.sum(), 1)


nt = ntrue.reindex(val_ids).to_numpy(dtype=np.int64)
print(f"\n{'th':>6} {'F05':>7} {'P':>7} {'R':>7} {'avgPred':>8} {'empty%':>7}")
for th in (0.80, 0.90, 0.93, 0.95, 0.97, 0.99):
    npd, tp = decide(th, True)
    f, p, r = f05_macro(npd, tp, nt)
    print(f"{th:>6.2f} {f:>7.4f} {p:>7.4f} {r:>7.4f} "
          f"{npd.sum()/len(val_ids):>8.2f} {100*(npd==0).mean():>6.1f}%")

npd, tp = decide(TH, True)
f, p, r = f05_macro(npd, tp, nt)
gs, ps = (nt == 0).sum(), (npd == 0).sum()
print(f"\nsingletons: true={gs:,} ({100*gs/len(val_ids):.1f}%)  "
      f"predicted={ps:,} ({100*ps/len(val_ids):.1f}%)")
print(f"singleton recall={((nt==0)&(npd==0)).sum()/max(gs,1):.4f}  "
      f"singleton precision={((nt==0)&(npd==0)).sum()/max(ps,1):.4f}")
print(f"\nFINAL: F05={f:.4f} P={p:.4f} R={r:.4f} "
      f"avgPred={npd.sum()/len(val_ids):.2f}")
