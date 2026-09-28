"""Phase 13 -- error analysis on the validation split, from the cached pairs."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import features as F  # noqa: E402
from eval_blocking import load_gt, make_split  # noqa: E402

TH = 0.95
val, fit = make_split()
val_ids = list(val[:30_000])
gt = load_gt(val_ids)
z = np.load(os.path.join(C.CACHE, "val_pairs.npz"), allow_pickle=True)
feats, names = z["feats"], list(z["feature_names"])
df = pd.DataFrame({"s1": z["s1_id"], "tg": z["tgt_id"], "src": z["tgt_src"],
                   "y": z["label"].astype(np.int32),
                   "p": z["prob_xgb"].astype(np.float64)})
sub = df[df.p >= TH]
fp = sub[sub.y == 0]
tp = sub[sub.y == 1]
print(f"predicted pairs={len(sub):,}  TP={len(tp):,}  FP={len(fp):,} "
      f"({100*len(fp)/len(sub):.2f}% of predictions)")

nfp = fp.groupby("s1").size()
ntp = tp.groupby("s1").size()
print(f"FP by source: S2={int((fp.src==2).sum()):,}  S3={int((fp.src==3).sum()):,}")
print(f"S1 with >=1 FP: {nfp.index.nunique():,}/{len(val_ids):,} "
      f"({100*nfp.index.nunique()/len(val_ids):.1f}%)")
print(f"FP per affected S1: mean={nfp.mean():.2f} max={nfp.max()}")

# where do the FPs sit in the candidate ranking?
fpi = np.flatnonzero((df.p >= TH).to_numpy() & (df.y == 0).to_numpy())
fi = feats[fpi]
cols = ["rank_n", "score", "score_ratio", "nm_ratio", "nm_tset", "nm_partial",
        "ad_ratio", "ad_tset", "num_ratio", "house_eq", "zip_eq", "mindf_min"]
have = [c for c in cols if c in names]
print("\nfeature means  FP vs TP (predicted pairs):")
tpi = np.flatnonzero((df.p >= TH).to_numpy() & (df.y == 1).to_numpy())
print(f"{'feature':>12} {'FP':>9} {'TP':>9}")
for c in have:
    k = names.index(c)
    print(f"{c:>12} {fi[:,k].mean():>9.3f} {feats[tpi][:,k].mean():>9.3f}")

# false negatives, split by cause: not retrieved at all, vs retrieved but scored
# below threshold. Look probabilities up in a dict -- filtering the frame once per
# pair is quadratic and takes minutes.
pmap = dict(zip(zip(df.s1, df.tg), df.p))
gt_pairs = {(e, t) for e, ts in gt.items() for t in ts}
absent = {pr for pr in gt_pairs if pr not in pmap}
below = {pr for pr in gt_pairs if pr in pmap and pmap[pr] < TH}
print(f"\ntrue pairs={len(gt_pairs):,}")
print(f"  not retrieved by blocking : {len(absent):,} "
      f"({100*len(absent)/len(gt_pairs):.2f}%)")
print(f"  retrieved but < th        : {len(below):,} "
      f"({100*len(below)/len(gt_pairs):.2f}%)")
print(f"  distinct S1 losing >=1    : {len({s for s, _ in below}):,}")
