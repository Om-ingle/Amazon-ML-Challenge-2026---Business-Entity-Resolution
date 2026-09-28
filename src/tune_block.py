"""Re-tune blocking against the current index/query builder.

Records the retrieval rank of every true pair, so candidate recall for any smaller
topk can be derived by truncation instead of re-running retrieval.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import blocking as B  # noqa: E402
import common as C  # noqa: E402
from eval_blocking import load_gt, make_split  # noqa: E402
from preprocess import load_norm  # noqa: E402

N = int(os.environ.get("N", "8000"))
CONFIGS = [
    dict(B.DEFAULTS, kq=100, budget=80_000, topk=30),
    dict(B.DEFAULTS, kq=100, budget=40_000, topk=30),
    dict(B.DEFAULTS, kq=60, budget=40_000, topk=30),
    dict(B.DEFAULTS, kq=100, budget=40_000, topk=30, w_addr=0.5),
]
TOPK_GRID = [3, 5, 10, 20, 30]

val, fit = make_split()
sub = list(val[:N])
gt = load_gt(sub)
ntrue = sum(len(v) for v in gt.values())
d1 = load_norm("train", 1)
pos = {e: i for i, e in enumerate(d1["entity_id"])}
idx = np.array([pos[e] for e in sub])
countries = d1["country"][idx]
print(f"queries={len(sub):,} true_pairs={ntrue:,} "
      f"countries={dict(zip(*np.unique(countries, return_counts=True)))}")

for p in CONFIGS:
    t0 = time.time()
    absent = 0
    ranks = []                      # rank of each retrieved true pair
    ncand = []
    for ctry in np.unique(countries):
        m = countries == ctry
        qids = d1["entity_id"][idx[m]]
        nmc, adn = d1["nmc"][idx[m]], d1["adn"][idx[m]]
        for src in (2, 3):
            ix = B.build_index("train", src, ctry)
            I, Sc, ptr = B.retrieve(nmc, adn, ix, p)
            ids = ix["ids"]
            ncand.extend(np.diff(ptr).tolist())
            for r, e in enumerate(qids):
                a, b = ptr[r], ptr[r + 1]
                truth = gt.get(e) or ()
                if not truth:
                    continue
                cand = ids[I[a:b]]
                rk = {t: j for j, t in enumerate(cand)}
                # matches contributed by the other source count as retrieved there;
                # here we only need this source's ranks, so absent-here is fine
                for t in truth:
                    j = rk.get(t)
                    if j is None:
                        absent += 1
                    else:
                        ranks.append(j)
            del ix
    ranks = np.asarray(ranks)
    el = time.time() - t0
    nc = np.asarray(ncand, float)
    # each true pair is searched in both source pools but only exists in its own,
    # so exactly ntrue of the "absent" observations are structurally impossible
    absent_true = (absent - ntrue) / ntrue
    line = (f"kq={p['kq']:4d} bud={p['budget']:7d} "
            f"cands/S1={2*nc.mean():6.1f} p95={2*np.percentile(nc,95):5.0f} "
            f"absent={absent_true:.4f} {el:6.0f}s | ")
    for k in TOPK_GRID:
        rec = (ranks < k).sum() / ntrue
        line += f"R@{k}={rec:.4f} "
    print(line, flush=True)
