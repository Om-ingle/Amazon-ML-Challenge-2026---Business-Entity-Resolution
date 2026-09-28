"""Phase 7 — pairwise match features.

All features are computed element-wise over arrays of pairs, with string metrics
batched through a process pool (rapidfuzz is C++ but called per string pair).

Feature families
  name      : char-level ratio / jaro-winkler / partial, token-set & token-sort
              ratios, sorted-unique-token ratio, length ratio & difference
  address   : same family on the address, plus numeric-token and ZIP/PIN views
  rarity    : minimum token document-frequency of each side (from the TF-IDF index)
  retrieval : TF-IDF score, rank within the candidate list, score margin to the
              best candidate for that S1, candidate-list size
  structural: script flags, token counts, target source indicator
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402
from rapidfuzz.distance import JaroWinkler  # noqa: E402

# Bump whenever FEATURE_NAMES or any metric definition changes: the pair cache is
# keyed on this, so a stale cache can never be mistaken for fresh features.
FEATURE_VERSION = "v2"

# (feature name, field-A key, field-B key, metric, scale)
RATIO_METRICS = [
    ("nm_ratio", "nm", "nm", "ratio", 100.0),
    ("nm_jw", "nm", "nm", "jw", 1.0),
    ("nm_tset", "nmc", "nmc", "token_set_ratio", 100.0),
    ("nm_tsort", "nmc", "nmc", "token_sort_ratio", 100.0),
    ("nm_partial", "nm", "nm", "partial_ratio", 100.0),
    ("nm_sorted", "nms", "nms", "ratio", 100.0),
    ("nm_wratio", "nm", "nm", "WRatio", 100.0),
    ("ad_ratio", "ad", "ad", "ratio", 100.0),
    ("ad_tset", "adn", "adn", "token_set_ratio", 100.0),
    ("ad_tsort", "adn", "adn", "token_sort_ratio", 100.0),
    ("ad_partial", "ad", "ad", "partial_ratio", 100.0),
    ("ad_sorted", "ads", "ads", "ratio", 100.0),
    ("ad_wratio", "ad", "ad", "WRatio", 100.0),
    ("num_ratio", "num", "num", "ratio", 100.0),
    ("zip_ratio", "zip", "zip", "ratio", 100.0),
    ("ad_jw", "ad", "ad", "jw", 1.0),
]

FIELDS = ["nm", "nmc", "nms", "ad", "adn", "ads", "num", "zip"]

_SCORERS = {
    "ratio": fuzz.ratio,
    "token_set_ratio": fuzz.token_set_ratio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "partial_ratio": fuzz.partial_ratio,
    "WRatio": fuzz.WRatio,
    "jw": JaroWinkler.similarity,
}

FEATURE_NAMES = (
    [m[0] for m in RATIO_METRICS]
    + ["nm_len_ratio", "nm_len_diff", "ad_len_ratio", "ad_len_diff",
       "nm_tok_ratio", "ad_tok_ratio", "mindf_a", "mindf_b", "mindf_min",
       "zip_eq", "zip_both", "house_eq",
       "score", "rank_n", "score_ratio", "score_margin", "ncand_n",
       "a_indic", "b_indic", "script_match", "ntok_nm_a", "ntok_nm_b",
       "ntok_ad_a", "ntok_ad_b", "is_s3", "a_name_empty", "b_name_empty",
       "a_addr_empty", "b_addr_empty"]
)


def _chunk_metrics(args):
    """Compute all string metrics for a slice of pairs. Runs in a worker process."""
    col = args
    out = {}
    for fname, ka, kb, metric, scale in RATIO_METRICS:
        A, B = col[(ka, "a")], col[(kb, "b")]
        if metric == "jw":
            out[fname] = np.fromiter(
                map(_SCORERS[metric], A, B), dtype=np.float32, count=len(A))
        else:
            out[fname] = np.fromiter(
                map(_SCORERS[metric], A, B), dtype=np.float32, count=len(A)) / scale
    return out


def _gather(rec, idx):
    return {f: rec[f][idx] for f in FIELDS}


def pair_features(a: dict, b: dict, ctx: dict, pool: Pool, chunk=20_000):
    """a, b: per-record dicts already gathered for each side of every pair.
    ctx: dict of per-pair numeric context arrays. Returns (P, F) float32."""
    n = len(ctx["score"])
    cols = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        col = {}
        for f in FIELDS:
            col[(f, "a")] = a[f][s:e].tolist()
            col[(f, "b")] = b[f][s:e].tolist()
        cols.append(col)
    parts = (pool.map(_chunk_metrics, cols) if pool is not None
             else [_chunk_metrics(c) for c in cols]) if cols else []
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} \
        if parts else {m[0]: np.empty(0, np.float32) for m in RATIO_METRICS}

    def ratio2(x, y):
        return np.where(np.maximum(x, y) > 0, np.minimum(x, y) / np.maximum(np.maximum(x, y), 1), 1.0)

    la, lb = a["_nm_len"], b["_nm_len"]
    aa, ab = a["_ad_len"], b["_ad_len"]
    out["nm_len_ratio"] = ratio2(la, lb)
    out["nm_len_diff"] = np.abs(la - lb)
    out["ad_len_ratio"] = ratio2(aa, ab)
    out["ad_len_diff"] = np.abs(aa - ab)
    out["nm_tok_ratio"] = ratio2(a["_ntn"], b["_ntn"])
    out["ad_tok_ratio"] = ratio2(a["_nta"], b["_nta"])
    out["mindf_a"] = np.log1p(a["_mindf"])
    out["mindf_b"] = np.log1p(b["_mindf"])
    out["mindf_min"] = np.log1p(np.minimum(a["_mindf"], b["_mindf"]))
    za, zb = a["zip"], b["zip"]
    out["zip_eq"] = (za == zb).astype(np.float32)
    out["zip_both"] = ((za != "") & (zb != "")).astype(np.float32)
    na, nb = a["num"], b["num"]
    out["house_eq"] = (na == nb).astype(np.float32)
    out["a_indic"] = a["_indic"]
    out["b_indic"] = b["_indic"]
    out["script_match"] = (a["_indic"] == b["_indic"]).astype(np.float32)
    out["ntok_nm_a"] = a["_ntn"]
    out["ntok_nm_b"] = b["_ntn"]
    out["ntok_ad_a"] = a["_nta"]
    out["ntok_ad_b"] = b["_nta"]
    for k in ("score", "rank_n", "score_ratio", "score_margin", "ncand_n", "is_s3"):
        out[k] = ctx[k]
    out["a_name_empty"] = (a["nm"] == "").astype(np.float32)
    out["b_name_empty"] = (b["nm"] == "").astype(np.float32)
    out["a_addr_empty"] = (a["ad"] == "").astype(np.float32)
    out["b_addr_empty"] = (b["ad"] == "").astype(np.float32)
    return np.stack([out[k] for k in FEATURE_NAMES], axis=1).astype(np.float32)


def decorate(rec: dict) -> dict:
    """Normalise the per-record scalar columns to their internal `_` names.

    The scalars are produced once by preprocess.py; this only renames them and
    attaches the index-derived rarity (`mindf`), so no per-block recomputation.
    """
    ren = {"nlen": "_nm_len", "alen": "_ad_len", "ntn": "_ntn",
           "nta": "_nta", "indic": "_indic"}
    for src, dst in ren.items():
        rec[dst] = np.asarray(rec[src]) if src in rec else _derive(rec, dst)
    rec["_mindf"] = np.asarray(rec.get("mindf",
                                       np.zeros(len(rec["nm"]), np.float32)))
    return rec


def _derive(rec: dict, dst: str):
    """Fallback when a record dict predates the cached scalar columns."""
    nm, ad, nmc, adn = rec["nm"], rec["ad"], rec["nmc"], rec["adn"]
    if dst == "_nm_len":
        return np.fromiter(map(len, nm), np.float32, len(nm))
    if dst == "_ad_len":
        return np.fromiter(map(len, ad), np.float32, len(ad))
    if dst == "_ntn":
        return np.fromiter((s.count(" ") + 1 if s else 0 for s in nmc),
                           np.float32, len(nmc))
    if dst == "_nta":
        return np.fromiter((s.count(" ") + 1 if s else 0 for s in adn),
                           np.float32, len(adn))
    return np.fromiter((1.0 if C.has_devanagari(s) else 0.0 for s in nm),
                       np.float32, len(nm))
