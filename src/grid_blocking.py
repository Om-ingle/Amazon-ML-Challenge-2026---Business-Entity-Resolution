"""Cheap parameter grid over blocking query budget (kq, budget, dfmax, topk).

Indices are cached, so each config costs only retrieval time.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import blocking as B  # noqa: E402
from eval_blocking import make_split, load_gt, eval_blocking  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
val, fit = make_split()
sub = val[:N]
gt = load_gt(sub)
print(f"n_s1={len(sub)} true_pairs={sum(len(v) for v in gt.values())}", flush=True)

GRID = []
for kq, budget in [(60, 40000), (100, 80000), (200, 200000)]:
    for topk in (20, 30, 45):
        GRID.append(dict(dfmax=100_000, budget=budget, kq=kq, topk=topk, w_addr=1.0))

rows = []
for p in GRID:
    t0 = time.time()
    _, _, met = eval_blocking("train", sub, gt, p, verbose=False)
    met["cfg"] = {k: p[k] for k in ("kq", "budget", "topk")}
    rows.append(met)
    print(f"kq={p['kq']:4d} budget={p['budget']:7d} topk={p['topk']:3d} | "
          f"candRecall={met['candidate_recall']:.4f} avgCand={met['avg_cands']:7.1f} "
          f"p95={met['p95_cands']:5d} max={met['max_cands']:5d} "
          f"zero={met['zero_cand_pct']:5.2f}% t={time.time()-t0:.0f}s", flush=True)

with open(os.path.join(os.path.dirname(__file__), "..",
                       "experiments", "block_grid.json"), "w") as f:
    json.dump(rows, f, indent=1)
print("saved")
