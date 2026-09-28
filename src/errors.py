"""Phase 13 — error analysis on the cached validation predictions.

Prints the actual records behind false positives and false negatives so the next
feature can be chosen from evidence rather than guesswork.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import features as F  # noqa: E402
from preprocess import load_norm  # noqa: E402

TH = float(os.environ.get("TH", "0.9"))
SHOW = int(os.environ.get("SHOW", "14"))

z = np.load(os.path.join(C.CACHE, "val_pairs.npz"), allow_pickle=True)
names = list(z["feature_names"])
X = z["feats"]
y = z["label"]
s1 = z["s1_id"]
tg = z["tgt_id"]
src = z["tgt_src"]
p = z["prob_xgb"]
fi = {n: i for i, n in enumerate(names)}
print(f"pairs={len(y):,} pos={y.sum():,} th={TH}")

pred = (p >= TH).astype(np.int8)
fp = np.flatnonzero((pred == 1) & (y == 0))
fn = np.flatnonzero((pred == 0) & (y == 1))
tp = np.flatnonzero((pred == 1) & (y == 1))
print(f"TP={len(tp):,} FP={len(fp):,} FN={len(fn):,}  "
      f"P={len(tp)/max(len(tp)+len(fp),1):.4f} R={len(tp)/max(len(tp)+len(fn),1):.4f}")

# per-true-pair recall by candidate rank: how much is blocking-ceiling vs threshold
rk = (X[:, fi["rank_n"]] * 30).astype(int)
sc = (X[:, fi["score_ratio"]])
if (y == 1).sum():
    print("\ntrue-pair recall by retrieval rank bucket (th applied):")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 31)]:
        m = (rk >= lo) & (rk < hi) & (y == 1)
        if m.sum() == 0:
            continue
        print(f"  rank [{lo:2d},{hi:2d}) n={m.sum():6d} meanScoreRatio={sc[m].mean():.3f} "
              f"recall_at_th={pred[m].mean():.3f}")
    print(f"  true pairs never retrieved (absent from candidates): "
          f"{'n/a (val set is candidate-only)'}")

# how many S1 have >=1 true pair below threshold
un = {}
for i in np.flatnonzero(y == 1):
    un.setdefault(s1[i], []).append(i)
ent_ok = ent_bad = 0
for k, v in un.items():
    if all(pred[i] for i in v):
        ent_ok += 1
    else:
        ent_bad += 1
print(f"\nS1 entities WITH >=1 true pair in candidates: {len(un):,}; "
      f"all true pairs above th: {ent_ok:,}; at least one missed: {ent_bad:,}")

# load strings for inspection
d1 = load_norm("train", 1)
d2 = load_norm("train", 2)
d3 = load_norm("train", 3)
pos1 = {e: i for i, e in enumerate(d1["entity_id"])}
pos2 = {e: i for i, e in enumerate(d2["entity_id"])}
pos3 = {e: i for i, e in enumerate(d3["entity_id"])}
d = {2: d2, 3: d3}
ps = {2: pos2, 3: pos3}


def show(idx, label):
    print(f"\n===== {label} (n={len(idx)}) — sorted by probability")
    order = idx[np.argsort(-p[idx])] if label.startswith("FP") else idx[np.argsort(p[idx])]
    for i in order[:SHOW]:
        r = ps[int(src[i])].get(tg[i])
        q = pos1.get(s1[i])
        if r is None or q is None:
            continue
        t = d[int(src[i])]
        print(f"  p={p[i]:.4f} score={X[i, fi['score']]:.3f} rank={rk[i]} "
              f"nm_r={X[i, fi['nm_ratio']]:.2f} ad_r={X[i, fi['ad_ratio']]:.2f} "
              f"nm_tset={X[i, fi['nm_tset']]:.2f} ad_tset={X[i, fi['ad_tset']]:.2f} "
              f"zip_eq={X[i, fi['zip_eq']]:.0f} ntf={X[i, fi['mindf_b']]:.2f}")
        print(f"      S1[{s1[i]}] {d1['nm'][q][:70]!r}")
        print(f"              {d1['ad'][q][:80]!r}")
        print(f"      {t['entity_id'][r][:9]} {t['nm'][r][:70]!r}")
        print(f"              {t['ad'][r][:80]!r}")


show(fp, "FALSE POSITIVES at th")
show(fn, "FALSE NEGATIVES at th")
