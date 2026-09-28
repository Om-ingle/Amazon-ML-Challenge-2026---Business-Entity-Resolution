# Results

**Amazon ML Challenge 2026 — Business Entity Resolution**

All figures below are measured. Nothing here is projected or estimated; where a
quantity was not measured, it is marked *not measured* instead of approximated.

---

## Final leaderboard

| metric | value |
|---|---|
| **F0.5** | **0.919** |

---

## Validation

Held-out, entity-level split of **30,000 Source-1 entities** from the training data.
The split is entity-level on Source 1 (no pair leakage), and the full Source-2/Source-3
pools remain searchable, so candidate lists contain the same kind of distractors as the
test set.

| metric | value |
|---|---|
| **F0.5 (macro, per Source-1 entity)** | **0.9334** |
| Precision (micro, over pairs) | 0.9738 |
| Recall (micro, over pairs) | 0.8811 |
| Validation Source-1 entities | 30,000 |
| Candidate recall (micro) | 0.9552 |
| Candidate recall (macro) | 0.9547 |
| Candidates per Source-1 entity (avg / median / p95 / max) | 60 / 60 / 60 / 60 |
| Entities receiving no candidate | 0.00% |
| Singleton recall / precision | 0.879 / 0.781 |
| True singleton rate | 5.4% |
| Predicted-empty rate | 6.0% |
| Candidate reduction ratio | 0.99998837 |

### Threshold sensitivity

| threshold | F0.5 | precision | recall | avg predicted / S1 | empty |
|---|---|---|---|---|---|
| 0.80 | 0.9171 | 0.9364 | 0.9185 | 3.40 | 4.6% |
| 0.90 | 0.9297 | 0.9587 | 0.9020 | 3.26 | 5.4% |
| 0.93 | 0.9326 | 0.9671 | 0.8919 | 3.20 | 5.7% |
| **0.95** | **0.9334** | **0.9738** | **0.8811** | **3.14** | **6.0%** |
| 0.97 | 0.9322 | 0.9819 | 0.8627 | 3.05 | 6.5% |
| 0.99 | 0.9197 | 0.9927 | 0.8150 | 2.85 | 7.7% |

The curve is flat on the precision side and steep on the recall side above 0.95, so
0.95 is a stable optimum rather than a knife-edge fit.

### Candidate recall vs. retrieval budget

| query posting budget | 20,000 | 40,000 | 80,000 | 200,000 |
|---|---|---|---|---|
| candidate recall | 0.9368 | 0.9495 | **0.9552** | 0.9570 |

80,000 is the knee: 200,000 costs 2.5× the runtime for +0.2pt.

Retrieval rank curve at the chosen budget — recall@3 = 0.8241, @5 = 0.8996,
@10 = 0.9299, @20 = 0.9473, @30 = 0.9552.

### Error analysis

| quantity | value |
|---|---|
| True pairs | 104,079 |
| Predicted pairs | 94,187 |
| True positives | 91,705 |
| False positives | 2,482 (2.64% of predictions) |
| True pairs retrieved but below threshold | 7.41% |
| True pairs never retrieved | 4.48% |

---

## Final test run

| quantity | value |
|---|---|
| Source-1 entities processed | 1,732,544 |
| — France | 259,452 |
| — India | 809,986 |
| — US | 663,106 |
| Candidate pairs | 103,952,559 |
| Matched pairs | 5,533,890 |
| Search pool (Source 2 + Source 3) | 9,969,589 |
| Candidates per Source-1 entity | 60 (median = p95 = max = 60) |
| Entities receiving no candidate | 0 (0.00%) |
| Predicted-empty rate | 5.91% |
| Avg matches per non-empty Source-1 entity | 3.39 |
| Candidate reduction ratio | 0.99999398 |

The predicted-empty rate is the load-bearing sanity number: it is what a broken
inference path distorts first, and it did — an earlier build with a chunk-local
indexing defect emitted 48.4% empty rows. At 5.91% against a validation expectation of
6.00%, with 3.39 matches per non-empty entity against 3.14 on validation, the test run
agrees with validation to within 8% on both.

---

## Why the validation and leaderboard scores differ

**0.9334 and 0.919 are different measurements and are not interchangeable.**

1. **An unseen country.** Validation is drawn from US and India only. The test split
   adds **France**, which never appears in the training data. France is handled by the
   same code path with no special-casing — `country` is an opaque string throughout, so
   nothing is hard-coded to a country list — but it is 15% of the test entities
   (259,452 of 1,732,544) and its accuracy is *not measured*, because the validation
   split contains no French entities to measure it on.
2. **Scale.** Validation is 30,000 entities; the test split is 1,732,544 — 58× larger
   and correspondingly more diverse.
3. **Sampling noise.** At 30,000 entities, differences below roughly 0.003 F0.5 were
   not treated as real when choosing between configurations.

The direction and rough size of the gap are therefore expected. The exact contribution
of France to it cannot be separated from this data, because no per-country test labels
are available — **not measured**.
