"""Phase 1+2 EDA: structure of the sources and of the ground truth.

Single pass, prints a compact report and writes cache/eda_summary.json.
Expensive per-pair statistics are computed on a fixed random sample.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

RNG = np.random.default_rng(0)
SAMPLE_S1 = 20000
rep = {}


def pct(x, n):
    return round(100.0 * x / max(n, 1), 2)


def field_stats(tag, d):
    n = len(d["entity_id"])
    name, addr, ctry = d["business_name"], d["business_address"], d["country"]
    nl = np.char.str_len(name.astype("U")) if False else np.array([len(s) for s in name])
    al = np.array([len(s) for s in addr])
    st = {
        "n": int(n),
        "empty_name": pct((nl == 0).sum(), n),
        "empty_addr": pct((al == 0).sum(), n),
        "name_len_mean": round(float(nl.mean()), 1),
        "name_len_p50": int(np.percentile(nl, 50)),
        "name_len_p95": int(np.percentile(nl, 95)),
        "addr_len_mean": round(float(al.mean()), 1),
        "addr_len_p50": int(np.percentile(al, 50)),
        "addr_len_p95": int(np.percentile(al, 95)),
        "countries": {k: int(v) for k, v in Counter(ctry).most_common()},
        "dup_entity_id": int(n - len(set(d["entity_id"]))),
    }
    # script mix per country on a sample (cheap enough on full data with regex)
    idx = RNG.choice(n, size=min(n, 200000), replace=False)
    sc = Counter((ctry[i], C.script_of(name[i])) for i in idx)
    st["name_script_by_country_pct"] = {
        f"{c}|{s}": pct(v, len(idx)) for (c, s), v in sc.most_common()
    }
    # exact duplicate (name, addr, country) rate, and normalized-name collisions
    keys = [C.alnum(name[i]) + "|" + C.alnum(addr[i]) + "|" + ctry[i] for i in idx]
    st["exact_dup_rate_sample"] = pct(len(idx) - len(set(keys)), len(idx))
    nkeys = [C.alnum(name[i]) for i in idx]
    st["norm_name_collision_rate_sample"] = pct(len(idx) - len(set(nkeys)), len(idx))
    rep[tag] = st
    print(f"\n[{tag}] n={n:,} emptyName={st['empty_name']}% emptyAddr={st['empty_addr']}% "
          f"dupId={st['dup_entity_id']}")
    print(f"   countries: {st['countries']}")
    print(f"   nameLen p50/p95={st['name_len_p50']}/{st['name_len_p95']} "
          f"addrLen p50/p95={st['addr_len_p50']}/{st['addr_len_p95']}")
    print(f"   script: {st['name_script_by_country_pct']}")
    print(f"   exactDup={st['exact_dup_rate_sample']}% normNameCollision="
          f"{st['norm_name_collision_rate_sample']}% (200k sample)")
    return st


print("=" * 70, "\nPHASE 1 — SOURCE FILES")
s1 = C.read_tsv(C.source_path("train", 1))
field_stats("train_s1", s1)
te1 = C.read_tsv(C.source_path("test", 1))
field_stats("test_s1", te1)

# ---------------------------------------------------------------- ground truth
print("=" * 70, "\nPHASE 2 — GROUND TRUTH")
gt = C.read_tsv(C.gt_path())
gs1, gm = gt["source1_entity_id"], gt["matched_entity_ids"]
print(f"gt rows={len(gs1):,} unique S1={len(set(gs1)):,} "
      f"S1-file rows={len(s1['entity_id']):,} "
      f"covers_all_s1={set(gs1) == set(s1['entity_id'])}")

n_m, n_s2, n_s3 = np.zeros(len(gs1), np.int32), np.zeros(len(gs1), np.int32), np.zeros(len(gs1), np.int32)
all_matched = []
for i, m in enumerate(gm):
    if not m:
        continue
    ids = m.split(",")
    n_m[i] = len(ids)
    c2 = sum(1 for x in ids if x[1] == "2")
    n_s2[i] = c2
    n_s3[i] = len(ids) - c2
    all_matched.extend(ids)

n = len(gs1)
cnt = Counter(n_m.tolist())
gtrep = {
    "n_s1": int(n),
    "zero_match_pct": pct((n_m == 0).sum(), n),
    "one_match_pct": pct((n_m == 1).sum(), n),
    "multi_match_pct": pct((n_m > 1).sum(), n),
    "match_count_hist": {int(k): int(v) for k, v in sorted(cnt.items())[:15]},
    "mean_matches": round(float(n_m.mean()), 3),
    "mean_matches_nonzero": round(float(n_m[n_m > 0].mean()), 3),
    "p95_matches": int(np.percentile(n_m, 95)),
    "max_matches": int(n_m.max()),
    "total_pairs": int(n_m.sum()),
    "s2_pairs": int(n_s2.sum()),
    "s3_pairs": int(n_s3.sum()),
    "has_s2_pct": pct((n_s2 > 0).sum(), n),
    "has_s3_pct": pct((n_s3 > 0).sum(), n),
    "n_s2_per_s1_nonzero": round(float(n_s2[n_s2 > 0].mean()), 3),
    "n_s3_per_s1_nonzero": round(float(n_s3[n_s3 > 0].mean()), 3),
    "matched_ids_total": len(all_matched),
    "matched_ids_unique": len(set(all_matched)),
}
rep["ground_truth"] = gtrep
print(json.dumps(gtrep, indent=1))

# country breakdown of match counts
c_of = dict(zip(s1["entity_id"], s1["country"]))
by_c = {}
for i in range(n):
    c = c_of.get(gs1[i], "?")
    a = by_c.setdefault(c, [0, 0, 0, 0])
    a[0] += 1
    a[1] += int(n_m[i] == 0)
    a[2] += int(n_s2[i])
    a[3] += int(n_s3[i])
rep["gt_by_country"] = {c: {"n_s1": a[0], "zero_pct": pct(a[1], a[0]),
                            "s2_per_s1": round(a[2] / a[0], 3),
                            "s3_per_s1": round(a[3] / a[0], 3)} for c, a in by_c.items()}
print("by country:", json.dumps(rep["gt_by_country"], indent=1))

# --------------------------------------------------- sampled true-pair analysis
print("=" * 70, "\nPHASE 2b — TRUE PAIR CHARACTERISTICS (sample)")
has = np.flatnonzero(n_m > 0)
samp = RNG.choice(has, size=min(SAMPLE_S1, len(has)), replace=False)
want = {}
for i in samp:
    for mid in gm[i].split(","):
        want[mid] = gs1[i]
s1row = {e: k for k, e in enumerate(s1["entity_id"])}

pair_stats = Counter()
sims = {"name_jac": [], "addr_jac": [], "name_alnum_eq": [], "addr_empty": []}
examples = []
for src in (2, 3):
    d = C.read_tsv(C.source_path("train", src))
    ids = d["entity_id"]
    keep = np.array([x in want for x in ids])
    ki = np.flatnonzero(keep)
    print(f"  matched rows found in S{src}: {len(ki):,}")
    for j in ki:
        s1id = want[ids[j]]
        i = s1row[s1id]
        n1, a1, c1 = s1["business_name"][i], s1["business_address"][i], s1["country"][i]
        n2, a2, c2 = d["business_name"][j], d["business_address"][j], d["country"][j]
        sc1, sc2 = C.script_of(n1), C.script_of(n2)
        pair_stats[f"S{src}|script_{sc1}->{sc2}"] += 1
        pair_stats[f"S{src}|country_eq_{c1 == c2}"] += 1
        pn1, pn2 = C.punct_norm(n1), C.punct_norm(n2)
        t1, t2 = set(C.name_core_tokens(pn1)), set(C.name_core_tokens(pn2))
        jac = len(t1 & t2) / max(len(t1 | t2), 1)
        pa1, pa2 = C.punct_norm(a1), C.punct_norm(a2)
        at1, at2 = set(C.addr_tokens(pa1)), set(C.addr_tokens(pa2))
        ajac = len(at1 & at2) / max(len(at1 | at2), 1)
        sims["name_jac"].append(jac)
        sims["addr_jac"].append(ajac)
        sims["name_alnum_eq"].append(C.alnum(n1) == C.alnum(n2))
        sims["addr_empty"].append(not a1 or not a2)
        pair_stats[f"S{src}|name_exact_alnum_{C.alnum(n1) == C.alnum(n2)}"] += 1
        pair_stats[f"S{src}|core_tok_share_{len(t1 & t2) > 0}"] += 1
        pair_stats[f"S{src}|addr_tok_share_{len(at1 & at2) > 0}"] += 1
        if len(examples) < 25 and jac < 0.5:
            examples.append((c1, n1, "||", n2, "//", a1, "||", a2))
    del d

tot_pairs = sum(len(sims["name_jac"]) for _ in [0])
arr = {k: np.array(v, dtype=float) for k, v in sims.items()}
ps = {
    "n_pairs_sampled": len(arr["name_jac"]),
    "name_core_jaccard_mean": round(float(arr["name_jac"].mean()), 3),
    "name_core_jaccard_p10": round(float(np.percentile(arr["name_jac"], 10)), 3),
    "name_core_jaccard_p50": round(float(np.percentile(arr["name_jac"], 50)), 3),
    "name_core_jaccard_eq0_pct": pct((arr["name_jac"] == 0).sum(), len(arr["name_jac"])),
    "addr_jaccard_mean": round(float(arr["addr_jac"].mean()), 3),
    "addr_jaccard_p50": round(float(np.percentile(arr["addr_jac"], 50)), 3),
    "addr_jaccard_eq0_pct": pct((arr["addr_jac"] == 0).sum(), len(arr["addr_jac"])),
    "name_alnum_exact_pct": pct(arr["name_alnum_eq"].sum(), len(arr["name_alnum_eq"])),
    "either_addr_empty_pct": pct(arr["addr_empty"].sum(), len(arr["addr_empty"])),
    "counts": {k: int(v) for k, v in pair_stats.most_common()},
}
rep["true_pairs"] = ps
print(json.dumps({k: v for k, v in ps.items() if k != "counts"}, indent=1))
print("counts:", json.dumps(ps["counts"], indent=1))
print("\n-- hard examples (name core jaccard < 0.5):")
for e in examples:
    print("   ", " ".join(str(x)[:60] for x in e))

with open(os.path.join(C.CACHE, "eda_summary.json"), "w", encoding="utf-8") as f:
    json.dump(rep, f, indent=1, ensure_ascii=False)
print("\nwrote cache/eda_summary.json")
