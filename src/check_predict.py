"""Validate predict.py's inference path against a known answer.

Runs exactly the same per-block routine predict.py uses, but on the TRAIN
validation ids, and recomputes macro F0.5. train.py scored 0.9334/0.9336 through
the pairs.py path, so agreement here means predict.py is faithful and any
difference seen on test is a property of the test data rather than a bug.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import features as F  # noqa: E402
from eval_blocking import load_gt, make_split  # noqa: E402
from predict import _run_block, load_model  # noqa: E402
from preprocess import load_norm  # noqa: E402

val, fit = make_split()
val_ids = list(val[:30_000])
gt = load_gt(val_ids)
model, cfg = load_model()
th = cfg["threshold"]
print(f"threshold={th} bij={cfg.get('bij')} val_f05 from train.py={cfg['val_f05']:.4f}")

d1 = F.decorate(load_norm("train", 1))
pos = {e: i for i, e in enumerate(d1["entity_id"])}
rows = np.array([pos[e] for e in val_ids])
countries = d1["country"][rows]

pred = {}
for ctry in np.unique(countries):
    cm = rows[countries == ctry]
    for src in (2, 3):
        cs, ms = _run_block((ctry, src), d1, cm, model, cfg, "train",
                            qchunk=int(os.environ.get("QCHUNK", "4000")),
                            verbose=False)
        for j, e in enumerate(d1["entity_id"][cm]):
            if ms[j]:
                pred.setdefault(e, set()).update(ms[j].split(","))

# bij=1: each target id may back only one S1 -> keep its best claimant
if cfg.get("bij"):
    best = {}
    for s, ts_ in pred.items():
        for t in ts_:
            if t not in best:
                best[t] = s
    keep = {}
    for s, ts_ in pred.items():
        for t in ts_:
            if best[t] == s:
                keep.setdefault(s, set()).add(t)
    pred = keep

m = C.macro_f05(pred, gt, val_ids)
n_empty = sum(1 for e in val_ids if not pred.get(e))
print(f"\nrecomputed via predict.py path: F05={m['f05_macro']:.4f} "
      f"P={m['micro_precision']:.4f} R={m['micro_recall']:.4f}")
print(f"predicted-empty={n_empty:,}/{len(val_ids):,} "
      f"({100*n_empty/len(val_ids):.1f}%)")
print(f"train.py reference         : F05={cfg['val_f05']:.4f} "
      f"P={cfg['val_precision']:.4f} R={cfg['val_recall']:.4f}")
print("VERDICT:", "MATCH" if abs(m["f05_macro"] - cfg["val_f05"]) < 0.01 else "MISMATCH")
