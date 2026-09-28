"""Smoke test: rebuild features for a small S1 subset and confirm the target-side
values match the cached strings. Cheap gate before the full pair rebuild."""
from __future__ import annotations

import os
import sys

import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import eval_blocking as EB  # noqa: E402
import pairs as P  # noqa: E402
from preprocess import load_norm  # noqa: E402

N = int(os.environ.get("N", "3000"))
sub = EB.load_gt(None)
ids = sorted(sub)[:N]
out = P.build_pairs("train", ids, gt=sub, pool=None, use_cache=False)

X = out["feats"]
fi = {n: i for i, n in enumerate(
    __import__("features").FEATURE_NAMES)}
s1, tg, src = out["s1_id"], out["tgt_id"], out["tgt_src"]
d1 = load_norm("train", 1)
d = {2: load_norm("train", 2), 3: load_norm("train", 3)}
p1 = {e: i for i, e in enumerate(d1["entity_id"])}
ps = {s: {e: i for i, e in enumerate(d[s]["entity_id"])} for s in (2, 3)}

rng = np.random.default_rng(0)
idx = rng.choice(len(s1), size=min(60, len(s1)), replace=False)
stats = {}
for f in ("nm_ratio", "ad_ratio", "nm_tset", "is_s3", "b_indic",
          "nm_len_diff", "b_addr_empty", "zip_eq", "house_eq"):
    bad = 0
    for i in idx:
        q = p1[s1[i]]
        r = ps[int(src[i])][tg[i]]
        t = d[int(src[i])]
        if f in ("nm_ratio", "ad_ratio", "nm_tset"):
            # metric definitions must mirror features.RATIO_METRICS exactly
            key = "nm" if f.startswith("nm") else "ad"
            ka, kb = ("nmc", "nmc") if f == "nm_tset" else (key, key)
            fn = {"nm_ratio": fuzz.ratio, "ad_ratio": fuzz.ratio,
                  "nm_tset": fuzz.token_set_ratio}[f]
            want = fn(d1[ka][q], t[kb][r]) / 100.0
        elif f == "is_s3":
            want = 1.0 if int(src[i]) == 3 else 0.0
        elif f == "b_indic":
            want = float(t["indic"][r])
        elif f == "nm_len_diff":
            want = float(abs(len(d1["nm"][q]) - len(t["nm"][r])))
        elif f == "b_addr_empty":
            want = float(t["ad"][r] == "")
        elif f == "zip_eq":
            # mirrors features.pair_features: equality only, empty==empty counts
            want = float(d1["zip"][q] == t["zip"][r])
        else:
            want = float(d1["num"][q] == t["num"][r])
        if abs(X[i, fi[f]] - want) > 1e-4 * max(1.0, abs(want)) + 1e-6:
            bad += 1
            if bad <= 3 and f in ("nm_ratio", "ad_ratio"):
                print(f"    MISMATCH {f} i={i} stored={X[i, fi[f]]:.4f} want={want:.4f}")
                print(f"      s1={s1[i]} {d1['nm'][q][:60]!r} | {d1['ad'][q][:60]!r}")
                print(f"      tg={tg[i]} {t['nm'][r][:60]!r} | {t['ad'][r][:60]!r}")
                print(f"      want_ratio_from_nm={fuzz.ratio(d1['nm'][q], t['nm'][r])/100:.4f}")
    stats[f] = bad
    print(f"  {f:14s} mismatches {bad}/{len(idx)}")

# label sanity on this subset
gt = sub
err = sum(1 for i in np.flatnonzero(out["label"] == 1)[:5000]
          if tg[i] not in gt.get(s1[i], ()))
print(f"label sanity: {err} positives not in gt")
print("pairs:", len(s1), "positives:", int(out["label"].sum()))
print("VERDICT:", "PASS" if sum(stats.values()) == 0 and err == 0 else "FAIL")
