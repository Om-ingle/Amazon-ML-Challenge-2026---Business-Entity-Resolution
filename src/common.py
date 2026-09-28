"""Shared IO + normalization utilities. Kept dependency-light and memory-conscious.

Design notes
------------
* Files are TSV with NO quoting: names/addresses contain raw `"` and `,` chars, so
  every reader must disable quote handling or fields get mangled.
* We use pyarrow.csv for speed, then hand back numpy object arrays (compact enough
  and much faster to index than pandas for our access patterns).
* Normalization is *multi-view*: we never collapse everything into one aggressive
  string. Each view is cheap to compute and cached to parquet.
"""
from __future__ import annotations

import os
import re
import unicodedata

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "dataset")
CACHE = os.path.join(ROOT, "cache")
EXP = os.path.join(ROOT, "experiments")
OUT = os.path.join(ROOT, "output")

# --------------------------------------------------------------------------- IO

_RO = pacsv.ReadOptions(block_size=1 << 26)
_PO = pacsv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False)


def read_tsv(path: str, columns=None) -> dict:
    """Read a challenge TSV into {column: numpy object array}."""
    co = pacsv.ConvertOptions(
        column_types={c: pa.string() for c in ("entity_id", "business_name",
                                               "business_address", "country",
                                               "source1_entity_id", "matched_entity_ids")},
        include_columns=columns,
        strings_can_be_null=False,
    )
    tbl = pacsv.read_csv(path, read_options=_RO, parse_options=_PO, convert_options=co)
    return {name: tbl.column(name).fill_null("").to_numpy(zero_copy_only=False)
            for name in tbl.column_names}


def source_path(split: str, src: int) -> str:
    return os.path.join(DATA, split, f"{split}_source{src}.tsv")


def gt_path() -> str:
    return os.path.join(DATA, "train", "train_ground_truth.tsv")


# ----------------------------------------------------------------- normalization

# Legal / organisational suffix tokens across US, India, France (+ generic).
LEGAL = {
    # generic / english
    "inc", "incorporated", "llc", "lc", "llp", "lp", "ltd", "limited", "co",
    "company", "corp", "corporation", "plc", "pllc", "pc", "pa", "trust",
    "holdings", "holding", "group", "enterprises", "enterprise", "ventures",
    "intl", "international",
    # india
    "pvt", "private", "pvtltd", "opc", "ltdco",
    # france
    "sarl", "sa", "sas", "sasu", "eurl", "sci", "snc", "scp", "scop", "sarlu",
    "ei", "eirl", "gie", "scm", "selarl", "sca", "scs",
    # other european that may leak in
    "gmbh", "ag", "bv", "nv", "srl", "spa", "oy", "ab", "as",
}

# Name token expansions (bidirectional canonicalisation to a single form).
NAME_ABBR = {
    "&": "and", "+": "and", "n": "and",
    "intl": "international", "int": "international", "natl": "national",
    "assoc": "associates", "assocs": "associates", "asso": "associates",
    "assn": "association", "bros": "brothers", "bro": "brothers",
    "mfg": "manufacturing", "mfrs": "manufacturers", "mfr": "manufacturers",
    "svc": "services", "svcs": "services", "serv": "services", "srvcs": "services",
    "sol": "solutions", "soln": "solutions", "solns": "solutions",
    "tech": "technologies", "techs": "technologies", "technology": "technologies",
    "ind": "industries", "inds": "industries", "industry": "industries",
    "engg": "engineering", "eng": "engineering", "engrs": "engineers",
    "constn": "construction", "cons": "construction",
    "dev": "development", "devs": "development",
    "mgmt": "management", "mgt": "management",
    "distr": "distributors", "dist": "distributors", "distrib": "distributors",
    "ent": "enterprises", "entp": "enterprises",
    "st": "saint", "ste": "sainte", "mt": "mount", "dr": "doctor",
    "univ": "university", "hosp": "hospital", "rest": "restaurant",
    "pharm": "pharmacy", "med": "medical", "lab": "laboratories",
    "labs": "laboratories", "laboratory": "laboratories",
    "auto": "automotive", "elec": "electric", "electricals": "electrical",
    "trdg": "trading", "exp": "exports", "imp": "imports",
    "agcy": "agency", "agencies": "agency",
    "prod": "products", "prods": "products", "product": "products",
    "sys": "systems", "system": "systems",
    "grp": "group", "ctr": "center", "centre": "center",
    "co-op": "cooperative", "coop": "cooperative",
}

# Address token canonicalisation (US + India + France common forms).
ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "blv": "boulevard", "dr": "drive", "ln": "lane",
    "ct": "court", "cir": "circle", "pl": "place", "plz": "plaza", "sq": "square",
    "hwy": "highway", "pkwy": "parkway", "pky": "parkway", "trl": "trail",
    "ter": "terrace", "tpke": "turnpike", "expy": "expressway", "frwy": "freeway",
    "ste": "suite", "apt": "apartment", "bldg": "building", "blk": "block",
    "fl": "floor", "flr": "floor", "rm": "room", "dept": "department",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "no": "number", "num": "number", "nos": "number", "opp": "opposite",
    "mkt": "market", "gali": "gali", "galli": "gali",
    "rly": "railway", "stn": "station", "jn": "junction", "ph": "phase",
    "sec": "sector", "ngr": "nagar", "col": "colony", "vill": "village",
    "po": "postoffice", "ps": "policestation", "dist": "district", "distt": "district",
    "teh": "tehsil", "tq": "taluk", "taluka": "taluk", "vpo": "village",
    "bh": "behind", "nr": "near", "flt": "flat", "gr": "ground", "grnd": "ground",
    # france
    "r": "rue", "bd": "boulevard", "av_": "avenue", "pl_": "place",
    "imp": "impasse", "all": "allee", "che": "chemin", "rte": "route",
    "sq_": "square", "res": "residence", "bat": "batiment", "app": "appartement",
    "zi": "zoneindustrielle", "za": "zoneartisanale", "zac": "zac", "cs": "cs",
    "bp": "boitepostale", "cedex": "cedex",
}

_WS = re.compile(r"\s+")
_NONALNUM = re.compile(r"[^0-9a-zऀ-ॿঀ-௿ఀ-ൿ]+")
_PUNCT = re.compile(r"[^0-9a-zऀ-ॿঀ-௿ఀ-ൿ\s]+")
_DIGITS = re.compile(r"\d+")
_DEV = re.compile(r"[ऀ-ॿ]")
_INDIC = re.compile(r"[ऀ-෿]")


def fold(s: str) -> str:
    """Unicode-fold + lowercase. Keeps Indic scripts; strips Latin accents."""
    if not s:
        return ""
    if s.isascii():            # fast path: most US records
        return s.lower()
    s = unicodedata.normalize("NFKD", s)
    # drop combining marks only for Latin (Indic vowel signs are significant, but
    # they live in the Indic blocks and NFKD leaves most of them as-is; we only
    # remove marks that follow ASCII letters).
    out = []
    for ch in s:
        if unicodedata.combining(ch):
            if out and out[-1].isascii():
                continue
            out.append(ch)
        else:
            out.append(ch)
    return "".join(out).lower()


def punct_norm(s: str) -> str:
    """Folded, punctuation -> space, whitespace collapsed."""
    return _WS.sub(" ", _PUNCT.sub(" ", fold(s))).strip()


def alnum(s: str) -> str:
    """Folded alphanumeric-only, no spaces."""
    return _NONALNUM.sub("", fold(s))


def name_tokens(pn: str) -> list:
    """Tokens of a punct-normalised name, abbreviation-canonicalised."""
    return [NAME_ABBR.get(t, t) for t in pn.split()]


def name_core_tokens(pn: str) -> list:
    """Name tokens with legal suffixes removed (never returns empty if input wasn't)."""
    toks = name_tokens(pn)
    core = [t for t in toks if t not in LEGAL]
    return core if core else toks


def addr_tokens(pa_: str) -> list:
    return [ADDR_ABBR.get(t, t) for t in pa_.split()]


def has_devanagari(s: str) -> bool:
    return bool(_DEV.search(s))


def script_of(s: str) -> str:
    if not s:
        return "empty"
    if _INDIC.search(s):
        return "indic"
    return "latin"


def char_ngrams(s: str, n: int = 3) -> list:
    if len(s) < n:
        return [s] if s else []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


# --------------------------------------------------------------------- metrics

def fbeta_half(tp: int, npred: int, ntrue: int) -> float:
    """Per-entity F0.5. Empty-truth + empty-pred scores 1.0 (challenge rule)."""
    if ntrue == 0 and npred == 0:
        return 1.0
    if npred == 0 or ntrue == 0:
        return 0.0
    p = tp / npred
    r = tp / ntrue
    if p == 0 and r == 0:
        return 0.0
    return (1.25 * p * r) / (0.25 * p + r)


def macro_f05(pred: dict, truth: dict, keys) -> dict:
    """Macro-averaged F0.5 over `keys` plus precision/recall/singleton diagnostics."""
    tot = tp_all = npred_all = ntrue_all = 0.0
    n = 0
    s_tp = s_fp = s_fn = s_tn = 0
    for k in keys:
        pr = pred.get(k) or ()
        tr = truth.get(k) or ()
        if not isinstance(pr, (set, frozenset)):
            pr = set(pr)
        if not isinstance(tr, (set, frozenset)):
            tr = set(tr)
        tp = len(pr & tr)
        tot += fbeta_half(tp, len(pr), len(tr))
        tp_all += tp
        npred_all += len(pr)
        ntrue_all += len(tr)
        n += 1
        if not tr:
            if not pr:
                s_tp += 1          # correctly predicted singleton
            else:
                s_fn += 1          # singleton we wrongly matched
        else:
            if not pr:
                s_fp += 1          # non-singleton we wrongly left empty
            else:
                s_tn += 1
    return {
        "f05_macro": tot / max(n, 1),
        "micro_precision": tp_all / max(npred_all, 1e-9),
        "micro_recall": tp_all / max(ntrue_all, 1e-9),
        "n_entities": n,
        "singleton_recall": s_tp / max(s_tp + s_fn, 1e-9),
        "singleton_precision": s_tp / max(s_tp + s_fp, 1e-9),
        "n_true_singletons": s_tp + s_fn,
        "n_pred_empty": s_tp + s_fp,
    }
