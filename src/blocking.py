"""Phase 5/6 — blocking: adaptive rare-token TF-IDF retrieval.

Architecture (per split, per target source, per country):
  * Inverted index  TT : (V x n_records) CSR, V = name-vocab | addr-vocab.
    Values are L2-normalised TF-IDF weights of the *target* record.
  * Query           Q  : (chunk x V) CSR built from the S1 record's tokens,
    keeping only the RAREST tokens subject to a hard posting-list budget.
    This bounds work per query -> no all-pairs comparison, predictable runtime.
  * Scores          S = Q @ TT, then top-k per row.

Country is a hard block (100% of training true pairs share country) and is treated
as an opaque string, so an unseen country (France) is handled like any other.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
from preprocess import load_norm  # noqa: E402

IDX = os.path.join(C.CACHE, "indexes")
# Bump when the index layout changes (vocabulary construction, weighting, ...):
# cached indexes are keyed on this so a stale layout can never be reused.
INDEX_VERSION = "i2"

DEFAULTS = dict(dfmax=100_000, budget=3000, kq=10, topk=20, w_addr=1.0)


# --------------------------------------------------------------------- indexing

def _explode(strings: np.ndarray):
    """Flatten token lists -> (codes, row_ids, vocab) with per-record dedup."""
    toks, rows = [], []
    for i, s in enumerate(strings):
        if not s:
            continue
        t = s.split()
        toks.extend(t)
        rows.extend([i] * len(t))
    if not toks:
        return (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, object))
    codes, vocab = pd.factorize(np.asarray(toks, dtype=object), sort=False)
    rows = np.asarray(rows, dtype=np.int64)
    key = rows * (len(vocab) + 1) + codes          # dedupe (record, token)
    key = np.unique(key)
    return key % (len(vocab) + 1), key // (len(vocab) + 1), vocab


def build_index(split: str, src: int, country: str) -> dict:
    tag = f"{split}_s{src}_{country.replace(' ', '_')}_{INDEX_VERSION}"
    f_npz = os.path.join(IDX, f"{tag}.npz")
    f_voc = os.path.join(IDX, f"{tag}_vocab.parquet")
    if os.path.exists(f_npz) and os.path.exists(f_voc):
        z = np.load(f_npz, allow_pickle=False)
        v = pq.read_table(f_voc)
        return dict(
            indptr=z["indptr"], indices=z["indices"], data=z["data"],
            df=z["df"], n=int(z["n"][0]), nvoc=int(z["n"][1]),
            mindf=z["mindf"], ntn=z["ntn"], nta=z["nta"],
            ids=pq.read_table(os.path.join(IDX, f"{tag}_ids.parquet"))
               .column("entity_id").to_numpy(zero_copy_only=False),
            tokens=v.column("token").to_numpy(zero_copy_only=False),
        )
    t0 = time.time()
    d = load_norm(split, src)
    m = d["country"] == country
    nmc, adn, ids = d["nmc"][m], d["adn"][m], d["entity_id"][m]
    n = len(ids)
    del d

    nc, nr, nvocab = _explode(nmc)
    ac, ar, avocab = _explode(adn)
    ntn = np.bincount(nr, minlength=n).astype(np.int16)
    nta = np.bincount(ar, minlength=n).astype(np.int16)
    # Names and addresses share one vocabulary (a word can occur in both), so the
    # two per-field vocabularies are merged into a single unique token space.
    n1 = len(nvocab)
    allv = np.concatenate([nvocab, avocab])
    uniq, inv = np.unique(allv, return_inverse=True)
    nc = inv[nc]
    ac = inv[ac + n1]
    del allv, inv, nvocab, avocab
    V = len(uniq)
    codes = np.concatenate([nc, ac])
    rows = np.concatenate([nr, ar])
    del nc, nr, ac, ar

    df = np.bincount(codes, minlength=V).astype(np.int64)
    idf = np.log(1.0 + n / np.maximum(df, 1)).astype(np.float32)
    w = idf[codes]
    norm = np.sqrt(np.bincount(rows, weights=w.astype(np.float64),
                               minlength=n))                     # per-record L2
    w = (w / np.maximum(norm[rows], 1e-9)).astype(np.float32)

    TT = sp.csr_matrix((w, (codes, rows)), shape=(V, n))
    TT.sum_duplicates()
    # per-record rarity scalars: rarest token df, and name/address token counts
    mindf = np.full(n, np.inf, np.float64)
    np.minimum.at(mindf, rows, df[codes].astype(np.float64))
    mindf = np.where(np.isfinite(mindf), mindf, 0.0).astype(np.float32)
    tokens = uniq
    np.savez(f_npz, indptr=TT.indptr, indices=TT.indices, data=TT.data,
             df=df, n=np.array([n, 0], np.int64),
             mindf=mindf, ntn=ntn, nta=nta)
    pq.write_table(pa.table({"token": pa.array(tokens.astype(object))}),
                   f_voc, compression="zstd")
    pq.write_table(pa.table({"entity_id": pa.array(ids)}),
                   os.path.join(IDX, f"{tag}_ids.parquet"), compression="zstd")
    print(f"  index {tag}: n={n:,} V={V:,} nnz={TT.nnz:,} {time.time()-t0:.0f}s")
    return dict(indptr=TT.indptr, indices=TT.indices, data=TT.data, df=df,
                n=n, nvoc=0, ids=ids, tokens=tokens,
                mindf=mindf, ntn=ntn, nta=nta)


def index_matrix(ix: dict) -> sp.csr_matrix:
    V = len(ix["df"])
    return sp.csr_matrix((ix["data"], ix["indices"], ix["indptr"]), shape=(V, ix["n"]))


# ---------------------------------------------------------------------- queries

def build_queries(nmc: np.ndarray, adn: np.ndarray, ix: dict, p: dict,
                  tok2id=None):
    """CSR query matrix using each record's rarest tokens under a posting budget.

    Fully vectorised: tokens are exploded, mapped to vocabulary ids, deduplicated,
    sorted by (record, df) and then truncated by a per-record cumulative posting
    budget. This is the same selection rule as a per-record loop but avoids Python
    per-record work, which dominates runtime at 1.7M+ queries.
    """
    V = len(ix["df"])
    df = ix["df"]
    nrec = ix["n"]
    dfmax, budget, kq, w_addr = p["dfmax"], p["budget"], p["kq"], p["w_addr"]
    if tok2id is None:
        tok2id = pd.Index(ix["tokens"]).get_indexer
    N = len(nmc)

    rows_l, codes_l, w_l = [], [], []
    for strings, w in ((nmc, 1.0), (adn, w_addr)):
        toks, rr = [], []
        for i, s in enumerate(strings):
            if s:
                t = s.split()
                toks.extend(t)
                rr.extend([i] * len(t))
        if not toks:
            continue
        codes = tok2id(np.asarray(toks, dtype=object))
        rr = np.asarray(rr, dtype=np.int64)
        ok = codes >= 0
        codes = codes[ok].astype(np.int64)
        rr = rr[ok]
        key = np.unique(rr * V + codes)            # dedupe (record, token)
        rows_l.append(key // V)
        codes_l.append(key % V)
        w_l.append(np.full(len(key), w, np.float32))
    if not rows_l:
        return sp.csr_matrix((N, V), dtype=np.float32)
    rows = np.concatenate(rows_l)
    codes = np.concatenate(codes_l)
    fw = np.concatenate(w_l)

    d = df[codes]
    order = np.lexsort((d, rows))               # by record, then rarest first
    rows, codes, fw, d = rows[order], codes[order], fw[order], d[order]

    gstart = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
    gsize = np.diff(np.r_[gstart, len(rows)])
    pos = np.arange(len(rows)) - np.repeat(gstart, gsize)
    excl = np.cumsum(d) - d                     # posting budget consumed so far
    excl = excl - np.repeat(excl[gstart], gsize)

    keep = (pos < kq) & (d <= dfmax) & (excl < budget)
    keep[pos == 0] = True                       # never starve a record entirely
    rows, codes, fw, d = rows[keep], codes[keep], fw[keep], d[keep]

    v = fw * np.log(1.0 + nrec / np.maximum(d, 1)).astype(np.float32)
    nrm = np.sqrt(np.bincount(rows, weights=v.astype(np.float64), minlength=N))
    v = (v / np.maximum(nrm[rows], 1e-9)).astype(np.float32)
    indptr = np.zeros(N + 1, np.int64)
    np.add.at(indptr, rows + 1, 1)
    np.cumsum(indptr, out=indptr)
    # rows come out grouped but not necessarily ascending; rebuild in row order
    o2 = np.argsort(rows, kind="stable")
    return sp.csr_matrix((v[o2], codes[o2], indptr), shape=(N, V))


def topk_from_scores(S: sp.csr_matrix, k: int):
    """Per-row top-k (indices, scores) from a CSR score matrix."""
    out_i, out_s, out_ptr = [], [], np.zeros(S.shape[0] + 1, np.int64)
    ip, ind, dat = S.indptr, S.indices, S.data
    for r in range(S.shape[0]):
        a, b = ip[r], ip[r + 1]
        if b == a:
            out_ptr[r + 1] = out_ptr[r]
            continue
        d = dat[a:b]
        if b - a > k:
            sel = np.argpartition(d, b - a - k)[b - a - k:]
        else:
            sel = np.arange(b - a)
        sc = d[sel]
        o = np.argsort(-sc, kind="stable")
        out_i.append(ind[a:b][sel][o])
        out_s.append(sc[o])
        out_ptr[r + 1] = out_ptr[r] + len(sel)
    if out_i:
        return np.concatenate(out_i), np.concatenate(out_s).astype(np.float32), out_ptr
    return np.empty(0, np.int32), np.empty(0, np.float32), out_ptr


def retrieve(nmc, adn, ix, p, qchunk=4000, verbose=False):
    """Return (cand_local_idx, scores, ptr) for the query records."""
    TT = index_matrix(ix)
    tok2id = pd.Index(ix["tokens"]).get_indexer   # once, not once per chunk
    k = p["topk"]
    I, Sc, ptr = [], [], np.zeros(len(nmc) + 1, np.int64)
    t0 = time.time()
    off = 0
    for a in range(0, len(nmc), qchunk):
        b = min(a + qchunk, len(nmc))
        Q = build_queries(nmc[a:b], adn[a:b], ix, p, tok2id)
        S = (Q @ TT).tocsr()
        i2, s2, p2 = topk_from_scores(S, k)
        I.append(i2)
        Sc.append(s2)
        ptr[a + 1:b + 1] = p2[1:] + off
        off += len(i2)
        if verbose and (a // qchunk) % 10 == 0:
            print(f"    {b}/{len(nmc)} nnzQ={Q.nnz} nnzS={S.nnz} "
                  f"{time.time()-t0:.0f}s", flush=True)
    return (np.concatenate(I) if I else np.empty(0, np.int32),
            np.concatenate(Sc) if Sc else np.empty(0, np.float32), ptr)
