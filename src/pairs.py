"""Build candidate pairs + features + labels for a set of Source-1 entities.

One pass per (country, target-source) block. Nothing is materialised for the whole
dataset at once, so peak memory stays bounded by the largest single block.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import blocking as B  # noqa: E402
import common as C  # noqa: E402
import features as F  # noqa: E402
from preprocess import load_norm  # noqa: E402

BLOCK = dict(B.DEFAULTS)
BLOCK.update(kq=100, budget=80_000, topk=30, dfmax=100_000)
SCALARS = ("nlen", "alen", "ntn", "nta", "indic")
# Bump on any change to the pairing/flattening logic below. Separate from
# F.FEATURE_VERSION, which covers only the feature *definitions*.
PAIRS_VERSION = "p2"


def _s1_side(d1, rows):
    """Gather the S1 records for the given local row indices."""
    return {f: d1[f][rows] for f in F.FIELDS}


def _cache_key(split, s1_ids, params, with_gt):
    h = hashlib.md5()
    h.update(split.encode())
    h.update(repr(sorted(params.items())).encode())
    h.update(F.FEATURE_VERSION.encode())
    h.update(PAIRS_VERSION.encode())
    h.update(b"|gt" if with_gt else b"|nogt")
    a = np.asarray(s1_ids, dtype=object)
    h.update(str(len(a)).encode())
    for x in a[:64]:
        h.update(str(x).encode())
    h.update(str(a[-1]).encode())
    return h.hexdigest()[:16]


def build_pairs(split, s1_ids, params=None, gt=None, pool=None,
                max_per_s1=None, verbose=True, use_cache=True):
    """Returns dict:
         s1_id   (P,)  object
         tgt_id  (P,)  object
         tgt_src (P,)  int8   (2 or 3)
         feats   (P,F) float32
         label   (P,)  int8 or None
         ncand   {s1_id: n}
    """
    p = dict(BLOCK if params is None else params)
    cf = os.path.join(C.CACHE, "candidate_pairs",
                      f"{_cache_key(split, s1_ids, p, gt is not None)}.npz")
    if use_cache and os.path.exists(cf):
        z = np.load(cf, allow_pickle=True)
        out = {k: z[k] for k in ("s1_id", "tgt_id", "tgt_src", "feats", "secs")}
        out["label"] = z["label"] if "label" in z else None
        out["ncand"] = {}
        if verbose:
            print(f"  [cached] {cf}  {len(out['s1_id']):,} pairs", flush=True)
        return out
    topk = p["topk"]
    d1 = load_norm(split, 1)
    pos = {e: i for i, e in enumerate(d1["entity_id"])}
    qrows = np.array([pos[e] for e in s1_ids])
    countries = d1["country"][qrows]

    o_s1, o_tg, o_src, o_feat, o_lab = [], [], [], [], []
    ncand = {}
    t0 = time.time()
    for ctry in np.unique(countries):
        cm = np.flatnonzero(countries == ctry)
        qids = d1["entity_id"][qrows[cm]]
        nmc, adn = d1["nmc"][qrows[cm]], d1["adn"][qrows[cm]]
        # Both sides must be restricted to THIS country block: `s1_row`/`t_row`
        # are local to the block, so indexing an all-query or all-record array
        # with them silently reads the wrong rows.
        k1 = {f: d1[f][qrows[cm]] for f in list(F.FIELDS) + list(SCALARS)}
        k1 = F.decorate(k1)
        for src in (2, 3):
            ix = B.build_index(split, src, ctry)
            ta = time.time()
            I, Sc, ptr = B.retrieve(nmc, adn, ix, p)
            ids = ix["ids"]
            # The index's record space is the COUNTRY-MASKED target records, so the
            # target dict must be restricted identically or `t_row` would address
            # the wrong records (this silently corrupted every target-side feature).
            tgt = load_norm(split, src, ctry)
            assert len(tgt["nm"]) == len(ix["ids"]) == ix["n"], "target mask mismatch"
            tgt["mindf"] = ix["mindf"]
            tgt = F.decorate(tgt)
            # ---- flatten the (query, candidate) lists into pair arrays
            lens = np.diff(ptr)
            s1_row = np.repeat(np.arange(len(qids)), lens)
            t_row = I.astype(np.int64)
            npair = int(lens.sum())
            base = np.zeros(len(lens), np.int64)
            np.cumsum(lens[:-1], out=base[1:])
            rank = np.arange(npair, dtype=np.int64) - np.repeat(base, lens)
            score = Sc.astype(np.float32)
            if len(score):
                # per-S1 best score / candidate count, computed on the score-desc order
                starts = ptr[:-1]
                best = score[starts[s1_row]]
                sr = np.divide(score, np.maximum(best, 1e-9))
                sm = best - score
                ncc = lens[s1_row].astype(np.float32)
            else:
                sr = sm = np.empty(0, np.float32)
                ncc = np.empty(0, np.float32)
            a = {f: k1[f][s1_row] for f in F.FIELDS}
            a.update({k: k1[k][s1_row] for k in
                      ("_nm_len", "_ad_len", "_ntn", "_nta", "_indic", "_mindf")})
            b = {f: tgt[f][t_row] for f in F.FIELDS}
            b.update({k: tgt[k][t_row] for k in
                      ("_nm_len", "_ad_len", "_ntn", "_nta", "_indic", "_mindf")})
            ctx = dict(score=score, rank_n=rank.astype(np.float32) / max(topk, 1),
                       score_ratio=sr.astype(np.float32),
                       score_margin=sm.astype(np.float32),
                       ncand_n=(ncc / max(2 * topk, 1)).astype(np.float32),
                       is_s3=np.full(len(score), 1.0 if src == 3 else 0.0,
                                     np.float32))
            feats = F.pair_features(a, b, ctx, pool)
            tids = ids[t_row]
            o_s1.append(np.repeat(qids, lens))
            o_tg.append(tids)
            o_src.append(np.full(len(tids), src, np.int8))
            o_feat.append(feats)
            if gt is not None:
                lab = np.array([1 if t in gt.get(s, ()) else 0
                                for s, t in zip(o_s1[-1], tids)], np.int8)
                o_lab.append(lab)
            # per-S1 candidate counts (max over sources) for diagnostics
            for i, k in enumerate(lens):
                ncand[qids[i]] = max(ncand.get(qids[i], 0), int(k))
            if verbose:
                print(f"    {ctry}|S{src}: q={len(qids):,} pairs={len(tids):,} "
                      f"{time.time()-ta:.0f}s", flush=True)
            del tgt, a, b, feats
        del nmc, adn
    out = dict(
        s1_id=np.concatenate(o_s1), tgt_id=np.concatenate(o_tg),
        tgt_src=np.concatenate(o_src), feats=np.concatenate(o_feat),
        ncand=ncand, secs=round(time.time() - t0, 1),
    )
    out["label"] = np.concatenate(o_lab) if o_lab else None
    if verbose:
        print(f"  build_pairs: {len(out['s1_id']):,} pairs in {out['secs']}s",
              flush=True)
    if use_cache:
        os.makedirs(os.path.dirname(cf), exist_ok=True)
        save = {k: v for k, v in out.items() if k != "ncand"}
        if save["label"] is None:
            save["label"] = np.empty(0, np.int8)
        np.savez(cf, **save)
    return out
