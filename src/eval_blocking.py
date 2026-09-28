"""Validation split + blocking evaluation (candidate recall / cost).

Split is entity-level on Source 1 (no pair leakage): a fixed random sample of S1
entities is held out for validation; the *full* S2/S3 pools stay searchable, which
mirrors test conditions (many distractor records).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import blocking as B  # noqa: E402
from preprocess import load_norm  # noqa: E402

VAL_N = 30_000
FIT_N = 150_000
SPLIT_F = os.path.join(C.CACHE, "split.npz")


def make_split():
    if os.path.exists(SPLIT_F):
        z = np.load(SPLIT_F, allow_pickle=True)
        return z["val"], z["fit"]
    d = load_norm("train", 1)
    n = len(d["entity_id"])
    rng = np.random.default_rng(7)
    perm = rng.permutation(n)
    val = d["entity_id"][perm[:VAL_N]]
    fit = d["entity_id"][perm[VAL_N:VAL_N + FIT_N]]
    np.savez(SPLIT_F, val=val, fit=fit)
    return val, fit


def load_gt(subset=None) -> dict:
    gt = C.read_tsv(C.gt_path())
    keep = set(subset) if subset is not None else None
    out = {}
    for s, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if keep is not None and s not in keep:
            continue
        out[s] = set(m.split(",")) if m else set()
    return out


def eval_blocking(split, s1_ids, gt, params, srcs=(2, 3), verbose=True):
    """Returns per-S1 candidate sets + metrics."""
    d1 = load_norm(split, 1)
    pos = {e: i for i, e in enumerate(d1["entity_id"])}
    idx = np.array([pos[e] for e in s1_ids])
    countries = d1["country"][idx]
    cands = {e: [] for e in s1_ids}
    scores = {e: [] for e in s1_ids}
    t0 = time.time()
    stats = {}
    for ctry in np.unique(countries):
        m = countries == ctry
        qi = idx[m]
        qids = d1["entity_id"][qi]
        nmc, adn = d1["nmc"][qi], d1["adn"][qi]
        for src in srcs:
            ix = B.build_index(split, src, ctry)
            ta = time.time()
            I, Sc, ptr = B.retrieve(nmc, adn, ix, params)
            ids = ix["ids"]
            for r, e in enumerate(qids):
                a, b = ptr[r], ptr[r + 1]
                if b > a:
                    cands[e].extend(ids[I[a:b]])
                    scores[e].extend(Sc[a:b])
            stats[f"{ctry}|S{src}"] = dict(nq=int(m.sum()),
                                           secs=round(time.time() - ta, 1))
            del ix
    el = time.time() - t0
    # metrics
    n_c = np.array([len(cands[e]) for e in s1_ids])
    tp = sum(len(set(cands[e]) & gt[e]) for e in s1_ids)
    tot = sum(len(gt[e]) for e in s1_ids)
    per_ent = [len(set(cands[e]) & gt[e]) / len(gt[e]) for e in s1_ids if gt[e]]
    pool = sum(B.build_index(split, s, c)["n"] for s in srcs for c in np.unique(countries))
    met = dict(
        candidate_recall=round(tp / max(tot, 1), 4),
        candidate_recall_macro=round(float(np.mean(per_ent)), 4),
        avg_cands=round(float(n_c.mean()), 1),
        med_cands=int(np.median(n_c)),
        p90_cands=int(np.percentile(n_c, 90)),
        p95_cands=int(np.percentile(n_c, 95)),
        max_cands=int(n_c.max()),
        zero_cand_pct=round(100.0 * (n_c == 0).mean(), 2),
        reduction_ratio=round(1 - n_c.mean() / (pool / max(len(np.unique(countries)), 1)), 8),
        secs=round(el, 1),
        secs_per_1k_s1=round(1000 * el / len(s1_ids), 2),
        per_block=stats,
    )
    if verbose:
        print(json.dumps({k: v for k, v in met.items() if k != "per_block"}, indent=1))
        print("  blocks:", stats)
    return cands, scores, met


if __name__ == "__main__":
    val, fit = make_split()
    print(f"val={len(val)} fit={len(fit)}")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    p = dict(B.DEFAULTS)
    for a in sys.argv[2:]:
        k, v = a.split("=")
        p[k] = float(v) if "." in v else int(v)
    print("params", p)
    sub = val[:n]
    gt = load_gt(sub)
    print(f"true pairs in subset: {sum(len(v) for v in gt.values())}")
    cands, scores, met = eval_blocking("train", sub, gt, p)
