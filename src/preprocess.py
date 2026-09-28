"""Phase 4 — multi-view normalization, computed once and cached to parquet.

Views stored per record:
  nm   punctuation/unicode-normalised name (lowercased, punct -> space)
  nmc  name core: abbreviation-canonicalised tokens with legal suffixes dropped
  ad   punctuation/unicode-normalised address
  adn  address tokens, abbreviation-canonicalised
Derived on the fly elsewhere (cheap): alnum = nm.replace(" ",""), tokens = split().

Parallelised with a process pool; each worker normalises a slice of strings.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

CHUNK = 250_000


_DIG = None


def _derived(nm, nmc, ad, adn):
    """Per-record views derived from the normalised forms (cheap, vectorised-ish).

    nms  sorted unique name-core tokens  -> reorder-robust name comparison
    ads  sorted unique address tokens    -> reorder-robust address comparison
    num  sorted distinct digit runs of the address (house no, PIN, floor ...)
    zip  sorted distinct digit runs of length >= 5 (US ZIP / India PIN / FR code)
    """
    import re as _re
    pat = _re.compile(r"\d+")
    nms, ads, num, zipc = [], [], [], []
    for c, d in zip(nmc, adn):
        nms.append(" ".join(sorted(set(c.split()))) if c else "")
        ads.append(" ".join(sorted(set(d.split()))) if d else "")
        m = pat.findall(d) if d else []
        m = sorted(set(m))
        num.append(" ".join(m))
        zipc.append(" ".join(sorted({x for x in m if len(x) >= 5})))
    return nms, ads, num, zipc


def _norm_chunk(args):
    names, addrs = args
    nm, nmc, ad, adn = [], [], [], []
    for s in names:
        p = C.punct_norm(s)
        nm.append(p)
        nmc.append(" ".join(C.name_core_tokens(p)) if p else "")
    for s in addrs:
        p = C.punct_norm(s)
        ad.append(p)
        adn.append(" ".join(C.addr_tokens(p)) if p else "")
    d = _derived(nm, nmc, ad, adn)
    # per-record scalars, so the feature stage never has to re-derive them
    nlen = [len(x) for x in nm]
    alen = [len(x) for x in ad]
    ntn = [x.count(" ") + 1 if x else 0 for x in nmc]
    nta = [x.count(" ") + 1 if x else 0 for x in adn]
    indic = [1 if C.has_devanagari(x) else 0 for x in names]
    return (nm, nmc, ad, adn) + d + (nlen, alen, ntn, nta, indic)


def normalize_file(split: str, src: int, ex: ProcessPoolExecutor) -> str:
    out = os.path.join(C.CACHE, "normalized", f"{split}_s{src}.parquet")
    if os.path.exists(out):
        print(f"  [cached] {out}")
        return out
    t0 = time.time()
    d = C.read_tsv(C.source_path(split, src))
    n = len(d["entity_id"])
    tasks = [(d["business_name"][i:i + CHUNK].tolist(),
              d["business_address"][i:i + CHUNK].tolist())
             for i in range(0, n, CHUNK)]
    nm, nmc, ad, adn, nms, ads, num, zipc = [], [], [], [], [], [], [], []
    nlen, alen, ntn, nta, indic = [], [], [], [], []
    for r in ex.map(_norm_chunk, tasks):
        nm += r[0]
        nmc += r[1]
        ad += r[2]
        adn += r[3]
        nms += r[4]
        ads += r[5]
        num += r[6]
        zipc += r[7]
        nlen += r[8]
        alen += r[9]
        ntn += r[10]
        nta += r[11]
        indic += r[12]
    tbl = pa.table({
        "entity_id": pa.array(d["entity_id"]),
        "country": pa.array(d["country"]).dictionary_encode(),
        "nm": pa.array(nm), "nmc": pa.array(nmc),
        "ad": pa.array(ad), "adn": pa.array(adn),
        "nms": pa.array(nms), "ads": pa.array(ads),
        "num": pa.array(num), "zip": pa.array(zipc),
        "nlen": pa.array(nlen, pa.int32()), "alen": pa.array(alen, pa.int32()),
        "ntn": pa.array(ntn, pa.int16()), "nta": pa.array(nta, pa.int16()),
        "indic": pa.array(indic, pa.int8()),
    })
    pq.write_table(tbl, out, compression="zstd", compression_level=3)
    print(f"  wrote {out}  n={n:,}  {time.time()-t0:.0f}s")
    return out


def load_norm(split: str, src: int, country: str | None = None) -> dict:
    """Load a normalized split. With `country`, only that country's rows are
    materialised -- the filter is applied in Arrow, so the other countries' strings
    are never converted to Python objects (this is what keeps per-block memory
    bounded when several blocks run concurrently)."""
    p = os.path.join(C.CACHE, "normalized", f"{split}_s{src}.parquet")
    t = (pq.read_table(p) if country is None
         else pq.read_table(p, filters=[("country", "=", country)]))
    return {c: (t.column(c).cast(pa.string()) if c == "country"
                else t.column(c)).to_numpy(zero_copy_only=False)
            for c in t.column_names}


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count())) as ex:
        for split in ("train", "test"):
            for src in (1, 2, 3):
                normalize_file(split, src, ex)
    print("done")
