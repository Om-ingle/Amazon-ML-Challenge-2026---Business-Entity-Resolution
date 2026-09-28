"""Phase 16/17 — test inference and output generation.

Work is split into (country, target-source) blocks. The heavy steps inside a block
-- the sparse TF-IDF product, the rapidfuzz metrics and the XGBoost predict -- are
C code that releases the GIL, so threads do run in parallel. Measured, though,
running both target sources concurrently is a net loss on a 16 GB machine: the
per-chunk sparse product of `Q @ TT` can reach `qchunk * budget` non-zeros (4000 x
80000), so two concurrent blocks peak well past available RAM and the run starts
swapping. `WORKERS=1` with a smaller `QCHUNK` keeps the footprint inside memory and
runs faster end to end. Country-filtered `load_norm` still bounds each block's
memory to one country's pools rather than the whole split.

Writes:
  output/candidate_pairs.tsv   the exact candidate set the final model scores
  output/matching_results.tsv  the final entity-level decisions
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import blocking as B  # noqa: E402
import common as C  # noqa: E402
import features as F  # noqa: E402
from preprocess import load_norm  # noqa: E402

EXP = os.path.join(C.CACHE, "..", "experiments")
DER = ("_nm_len", "_ad_len", "_ntn", "_nta", "_indic", "_mindf")


def _rss_gb():
    """Current working set in GB, for progress logging only.

    Deliberately dependency-free: the run this reports on is the one place where a
    memory stall is invisible in wall-clock terms, so the diagnostic must not be
    able to fail on an import. Falls back to NaN anywhere it cannot measure."""
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        c = _PMC()
        c.cb = ctypes.sizeof(_PMC)
        # argtypes matter here: without them ctypes passes GetCurrentProcess()'s
        # pseudo-handle as a 32-bit int, the truncated handle is rejected, and the
        # call fails while leaving the struct zeroed -- which reads as "0 GB" rather
        # than as an error.
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        if not fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return float("nan")
        return c.WorkingSetSize / 2 ** 30
    except Exception:
        return float("nan")


def load_model():
    with open(os.path.join(EXP, "best", "model.pkl"), "rb") as f:
        m = pickle.load(f)
    with open(os.path.join(EXP, "best", "config.json")) as f:
        cfg = json.load(f)
    return m, cfg


def score_chunk(X, m, cfg):
    p = m["xgb"].predict_proba(X)[:, 1].astype(np.float32)
    w = cfg.get("blend_lr", 0.0)
    if w:
        pl = m["lr"].predict_proba(m["scaler"].transform(X))[:, 1].astype(np.float32)
        p = (1.0 - w) * p + w * pl
    return p


def _block_cache_path(split, ctry, src):
    return os.path.join(C.CACHE, "predict_blocks",
                        f"{split}_{ctry}_S{src}.npz".replace(" ", "_"))


def _run_block(task, d1, cm, model, cfg, split, qchunk, verbose, collect=None,
               use_cache=True):
    """Score one (country, source) block. Returns (cand_strs, match_strs) for the
    S1 entities in `cm`, in the order they appear in `cm`.

    Results are cached per block, so an interrupted full-test run resumes from the
    last completed block instead of restarting (the full run is hours long).

    `collect`, when a list is passed, receives one (s1_ids, tgt_ids, feats, prob)
    tuple per chunk. Used only by the verification harness -- it lets a caller
    diff this path's pairs and features against pairs.py on identical input."""
    ctry, src = task
    cfile = _block_cache_path(split, ctry, src)
    if use_cache and collect is None and os.path.exists(cfile):
        z = np.load(cfile, allow_pickle=True)
        if len(z["cm"]) == len(cm) and np.array_equal(z["cm"], cm):
            print(f"  {ctry}|S{src} [cached] {len(cm):,} S1", flush=True)
            return list(z["cs"]), list(z["ms"])
    params, th = cfg["blocking"], cfg["threshold"]
    topk = params["topk"]
    nc = len(cm)
    cs, ms = [""] * nc, [""] * nc
    q_nmc, q_adn = d1["nmc"][cm], d1["adn"][cm]
    ix = B.build_index(split, src, ctry)
    # `t_row` is local to this country block, so the target arrays must be
    # restricted to the same block -- indexing un-restricted arrays with it
    # silently reads the wrong records.
    tgt = load_norm(split, src, ctry)
    assert len(tgt["nm"]) == len(ix["ids"]) == ix["n"], "target mask mismatch"
    tgt["mindf"] = ix["mindf"]
    tgt = F.decorate(tgt)
    idstr = ix["ids"].astype(str)
    ta = time.time()
    tlast = ta
    for a in range(0, nc, qchunk):
        b = min(a + qchunk, nc)
        I, Sc, ptr = B.retrieve(q_nmc[a:b], q_adn[a:b], ix, params)
        lens = np.diff(ptr)
        s1_row = np.repeat(np.arange(b - a), lens)
        t_row = I.astype(np.int64)
        base = np.zeros(len(lens), np.int64)
        np.cumsum(lens[:-1], out=base[1:])
        rank = np.arange(int(lens.sum()), dtype=np.int64) - np.repeat(base, lens)
        score = Sc.astype(np.float32)
        best = score[ptr[:-1][s1_row]]
        ctx = dict(
            score=score,
            rank_n=rank.astype(np.float32) / max(topk, 1),
            score_ratio=np.divide(score, np.maximum(best, 1e-9), dtype=np.float32),
            score_margin=(best - score).astype(np.float32),
            ncand_n=(lens[s1_row] / max(2 * topk, 1)).astype(np.float32),
            is_s3=np.full(len(score), 1.0 if src == 3 else 0.0, np.float32))
        # `s1_row` is local to this chunk, so it has to be mapped back through `cm`
        # to address the S1 records: using it directly against a dataset-spanning
        # array silently re-reads the FIRST chunk's S1 records for every later chunk
        # (and only for chunks after the first, so a single-chunk smoke test cannot
        # catch it). Gathering per chunk also avoids materialising a copy of the
        # whole country's S1 fields, which for India is a large share of the memory
        # budget and was pushing the run into swap.
        grow = cm[a + s1_row]
        A = {f: d1[f][grow] for f in F.FIELDS}
        A.update({k: np.asarray(d1[k])[grow] for k in DER})
        Bs = {f: tgt[f][t_row] for f in F.FIELDS}
        Bs.update({k: tgt[k][t_row] for k in DER})
        X = F.pair_features(A, Bs, ctx, None)
        p = score_chunk(X, model, cfg)
        if collect is not None:
            collect.append((
                np.repeat(np.asarray(d1["entity_id"][cm[a:b]], dtype=object), lens),
                idstr[t_row].astype(object), X, p))
        del X, A, Bs, ctx
        for r in range(b - a):
            s, e = ptr[r], ptr[r + 1]
            if e == s:
                continue
            g = a + r
            ids_r = idstr[t_row[s:e]]
            cs[g] = ",".join(ids_r)
            k = p[s:e] >= th
            if k.any():
                ms[g] = ",".join(ids_r[k])
        # Time-based, not chunk-count-based: a chunk-count interval goes silent for
        # hours exactly when the run is slowest, which is when the progress signal
        # matters most. Reporting the rate and working set makes a memory stall
        # visible instead of looking like a hang.
        now = time.time()
        if verbose and now - tlast >= 30.0:
            tlast = now
            el = now - ta
            print(f"    {ctry}|S{src} {b}/{nc} {el:.0f}s "
                  f"({b/el:.1f} q/s, rss={_rss_gb():.1f}GB)", flush=True)
    del tgt, idstr, ix
    print(f"  {ctry}|S{src} done {time.time()-ta:.0f}s", flush=True)
    if use_cache and collect is None:
        os.makedirs(os.path.dirname(cfile), exist_ok=True)
        np.savez(cfile, cm=cm, cs=np.array(cs, dtype=object),
                 ms=np.array(ms, dtype=object))
    return cs, ms


def run_blocks(split="test", countries=None, qchunk=4000, workers=2,
               verbose=True):
    """Compute (and cache) every (country, source) block.

    Countries are processed one at a time so peak memory is bounded by a single
    country's S1 slice plus its two target pools, rather than the whole split.
    Blocks that already have a cache entry are skipped, which makes an interrupted
    multi-hour run resumable.
    """
    model, cfg = load_model()
    import pyarrow as pa
    import pyarrow.parquet as pq
    _c = pq.read_table(os.path.join(C.CACHE, "normalized", f"{split}_s1.parquet"),
                       columns=["country"]).column("country").cast(pa.string())
    ctry_all = list(np.unique(_c.to_numpy(zero_copy_only=False)))
    if countries is not None:
        ctry_all = [c for c in ctry_all if c in set(countries)]
    t0 = time.time()
    print(f"{split}: countries={ctry_all} blocking={cfg['blocking']} "
          f"th={cfg['threshold']} workers={workers}", flush=True)
    for ctry in ctry_all:
        d1 = F.decorate(load_norm(split, 1, ctry))
        cm = np.arange(len(d1["entity_id"]))
        print(f"  {ctry}: {len(cm):,} S1 ({time.time()-t0:.0f}s)", flush=True)
        # build the two indexes sequentially: they are shared read-only by the
        # threads below and a concurrent first build would multiply peak memory
        for src in (2, 3):
            B.build_index(split, src, ctry)
        tasks = [(ctry, 2), (ctry, 3)]
        with ThreadPoolExecutor(max_workers=max(1, min(workers, 2))) as ex:
            futs = [ex.submit(_run_block, t, d1, cm, model, cfg, split,
                              qchunk, verbose) for t in tasks]
            for f in futs:
                f.result()
        del d1
    print(f"all blocks done ({time.time()-t0:.0f}s)", flush=True)


def assemble(split="test", out_dir=None):
    """Merge the cached blocks into the two output TSVs."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    out_dir = out_dir or C.OUT
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(C.CACHE, "normalized", f"{split}_s1.parquet")
    t = pq.read_table(p, columns=["entity_id", "country"])
    ids = t.column("entity_id").to_numpy(zero_copy_only=False)
    ctry = t.column("country").cast(pa.string()).to_numpy(zero_copy_only=False)
    n = len(ids)
    cand_str = np.array([""] * n, dtype=object)
    match_str = np.array([""] * n, dtype=object)
    for c in np.unique(ctry):
        m = np.flatnonzero(ctry == c)
        for src in (2, 3):
            z = np.load(_block_cache_path(split, c, src), allow_pickle=True)
            cm, cs, ms = z["cm"], z["cs"], z["ms"]
            if not (len(cm) == len(m) and np.array_equal(cm, np.arange(len(m)))):
                raise RuntimeError(f"block {c}|S{src} does not cover its country")
            for j, base in enumerate(m):
                if cs[j]:
                    cand_str[base] = cs[j] if not cand_str[base] \
                        else cand_str[base] + "," + cs[j]
                if ms[j]:
                    match_str[base] = ms[j] if not match_str[base] \
                        else match_str[base] + "," + ms[j]
    mp = os.path.join(out_dir, "matching_results.tsv")
    cp = os.path.join(out_dir, "candidate_pairs.tsv")
    with open(mp, "w", encoding="utf-8", newline="\n") as fm, \
         open(cp, "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for i in range(n):
            fm.write(f"{ids[i]}\t{match_str[i]}\n")
            fc.write(f"{ids[i]}\t{cand_str[i]}\n")
    print(f"wrote {mp} and {cp}  ({n:,} rows)")
    return mp, cp


def predict_split(split="test", out_dir=None, limit=None, qchunk=4000,
                  workers=2, verbose=True):
    out_dir = out_dir or C.OUT
    os.makedirs(out_dir, exist_ok=True)
    if limit is not None:
        # a bounded smoke run: score a prefix of each country without touching the
        # block cache, so a partial run can never be mistaken for a complete one
        model, cfg = load_model()
        d1 = F.decorate(load_norm(split, 1))
        countries = d1["country"]
        ctry_list = list(np.unique(countries))
        tasks, cms = [], []
        for ctry in ctry_list:
            c = np.flatnonzero(countries == ctry)
            take = max(1, int(round(limit * len(c) / len(countries))))
            for src in (2, 3):
                tasks.append((ctry, src))
                cms.append(c[:take])
        for ctry in ctry_list:
            for src in (2, 3):
                B.build_index(split, src, ctry)
        results = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as ex:
            futs = {ex.submit(_run_block, t, d1, cm, model, cfg, split, qchunk,
                              verbose, None, False): i
                    for i, (t, cm) in enumerate(zip(tasks, cms))}
            for f, i in futs.items():
                results[i] = f.result()
        return results
    run_blocks(split=split, qchunk=qchunk, workers=workers, verbose=verbose)
    return assemble(split=split, out_dir=out_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "assemble":
        assemble()
    elif len(sys.argv) > 1 and sys.argv[1] == "blocks":
        wk = int(os.environ.get("WORKERS", "2"))
        qc = int(os.environ.get("QCHUNK", "4000"))
        cs = sys.argv[2].split(",") if len(sys.argv) > 2 else None
        run_blocks(countries=cs, workers=wk, qchunk=qc)
    else:
        predict_split(workers=int(os.environ.get("WORKERS", "2")))
