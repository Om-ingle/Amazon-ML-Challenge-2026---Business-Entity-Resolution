"""Correctness check: does the stored feature matrix actually correspond to the
stored (s1_id, tgt_id) pair? Recomputes a few features from the cached strings and
compares against the values in cache/val_pairs.npz."""
from __future__ import annotations

import os
import sys

import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
from preprocess import load_norm  # noqa: E402

z = np.load(os.path.join(C.CACHE, "val_pairs.npz"), allow_pickle=True)
names = list(z["feature_names"])
fi = {n: i for i, n in enumerate(names)}
X, y = z["feats"], z["label"]
s1, tg, src = z["s1_id"], z["tgt_id"], z["tgt_src"]
print("features:", names)
print("X dtype/shape", X.dtype, X.shape, "nan:", np.isnan(X).sum(),
      "inf:", np.isinf(X).sum())

d1 = load_norm("train", 1)
d = {2: load_norm("train", 2), 3: load_norm("train", 3)}
p1 = {e: i for i, e in enumerate(d1["entity_id"])}
ps = {s: {e: i for i, e in enumerate(d[s]["entity_id"])} for s in (2, 3)}

rng = np.random.default_rng(0)
idx = rng.choice(len(y), size=40, replace=False)
bad = 0
for i in idx:
    q = p1[s1[i]]
    r = ps[int(src[i])][tg[i]]
    for fname, key, scale in (("nm_ratio", "nm", 100.0), ("ad_ratio", "ad", 100.0)):
        got = X[i, fi[fname]] * scale
        want = fuzz.ratio(d1[key][q], d[int(src[i])][key][r])
        if abs(got - want) > 1.0:
            bad += 1
            print(f"  MISMATCH i={i} {fname}: stored={got:.1f} recomputed={want:.1f}")
            print(f"    s1={s1[i]} {d1['nm'][q][:50]!r}")
            print(f"    tg={tg[i]} {d[int(src[i])]['nm'][r][:50]!r}")
print(f"checked 40 rows, mismatches={bad}")

# label sanity: every label==1 must have tgt in ground truth of that s1
gt = {}
import csv
with open(C.gt_path(), encoding="utf-8") as f:
    f.readline()
    for line in f:
        a, _, b = line.partition("\t")
        gt[a] = set(b.rstrip("\n").split(",")) if b.strip() else set()
pos = np.flatnonzero(y == 1)
err = sum(1 for i in pos[:5000] if tg[i] not in gt.get(s1[i], ()))
print(f"label sanity: {err} / 5000 positives whose target is NOT in ground truth")
# and the reverse: are all true pairs present as positives?
sub = np.unique(s1)
miss = 0
tot = 0
for e in sub[:2000]:
    have = set(tg[(s1 == e)])
    for t in gt.get(e, ()):
        tot += 1
        if t not in have:
            miss += 1
print(f"coverage: {miss} / {tot} true pairs missing from the candidate feature rows")
