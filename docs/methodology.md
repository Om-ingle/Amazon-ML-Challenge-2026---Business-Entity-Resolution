# Methodology

Amazon ML Challenge 2026 — Business Entity Resolution.
Companion documents: [../README.md](../README.md), [../results/results.md](../results/results.md),
[../configs/best_config.json](../configs/best_config.json).

---

## 1. Problem definition

Three sources of business records are provided. **Source 1** is the reference entity
set; **Source 2** and **Source 3** are noisy pools containing the same underlying
businesses. For every Source-1 entity the task is to return the set of Source-2 and
Source-3 record ids that refer to the same real-world business, or an empty set when
none do.

The scoring metric is **macro-averaged F0.5 (β = 0.5)**, computed per Source-1 entity
and then averaged. Because β < 1, precision is weighted more heavily than recall:

```
F0.5 = (1.25 × P × R) / (0.25 × P + R)
```

Predicting an empty match set for a true singleton is scored as correct and earns full
credit for that entity. Predicting empty when matches exist forfeits the entity
entirely, so a conservative threshold is favoured but matches are never forced.

An entity-level assignment is imposed by the ground truth: every matched Source-2/3 id
appears exactly once, i.e. a target record backs at most one Source-1 entity.

---

## 2. Data structure

Each record carries an entity id, a business name, a free-text address, and a
`country` field. Country is treated as an **open-set string** rather than a fixed
enum — nothing in the pipeline is specialized to US or India, so the unseen test
country (France) is handled by exactly the same code path.

Observations that shaped the design (from `src/eda.py`):

- **Country is a perfect blocking key.** 100% of training true pairs share a country.
- **Records are short.** Most names are 2–5 tokens; most addresses are under 12.
  Character-level similarity is therefore informative and cheap.
- **Positives are rare among candidates.** Only ~5% of retrieved candidate pairs are
  true matches, which is what makes the precision-weighted metric the right target.
- **The singleton rate is low.** 5.4% of validation Source-1 entities have no match at
  all, and the average entity has 3.47 true matches — so predicting empty is rarely
  correct, but it is scored.

A data-handling detail that matters in practice: the TSVs have **no quoting**.
Business names and addresses contain raw `"` and `,` characters, so every reader must
disable quote handling or fields are silently mangled. `common.read_tsv` uses
`pyarrow.csv` with `quote_char=False`.

---

## 3. Data normalization

Normalization is **multi-view**: rather than collapsing each record into one
aggressively normalized string (which destroys evidence), several views are computed
once and cached to parquet. Each view is cheap and each downstream feature picks the
view that suits it.

| view | purpose |
|---|---|
| `nm`, `ad` | Unicode-folded, lowercased, punctuation-normalized |
| `nmc`, `adn` | abbreviation-canonicalized, legal-suffix-stripped core tokens |
| `nms`, `ads` | sorted unique tokens (reorder-robust) |
| `num` | distinct digit runs |
| `zip` | long digit runs (ZIP / PIN / house-number evidence) |
| `_indic` | Indian-script detection flag |

Caching matters here: the full test split is 1.73M Source-1 records plus a 9.97M-record
pool, and every stage downstream re-reads these views. Computing them once and storing
parquet keeps re-runs cheap.

---

## 4. Name / address canonicalization

Two canonicalization steps carry most of the weight.

**Legal-suffix stripping.** Organisational suffixes (`Pvt Ltd`, `LLC`, `LLP`, `Inc`,
`SARL`, `GmbH`, …) are removed from the name core. The suffix set is deliberately
generic and covers US, India, and France, since a suffix-only difference is never
evidence of a different entity.

**Abbreviation canonicalization.** Common business abbreviations are expanded
(`Pvt` → `Private`, `Intl` → `International`, `Corp` → `Corporation`, …) so that
abbreviated and expanded forms of the same name converge.

Both operate on token lists, and the result (`nmc`/`adn`) feeds the token-set and
token-sort features. A separate sorted-unique-token view (`nms`/`ads`) handles pure
token reordering, which is common in the address field.

---

## 5. Country-aware blocking

Country is applied as a **hard block**: candidate retrieval never crosses a country
boundary. This is justified by the training data (100% of true pairs share a country)
and it bounds the search space before any similarity work happens.

Critically, the country value is used **opaquely**. There is no list of known
countries, no special case for any country, and no fallback logic keyed on a country
name. Adding France to the test split required no code change, which is the open-set
requirement satisfied structurally rather than by assertion.

---

## 6. TF-IDF retrieval

Within a country, an inverted index is built over a **merged vocabulary** of
name-core and address-core tokens, stored as a sparse `V × n_records` CSR matrix of
**L2-normalized IDF weights**.

Scoring a query is a sparse product `Q @ TT`, followed by per-row top-k over both
target pools. Because the matrix is normalized, an inner product behaves like a cosine
similarity over weighted tokens: a match on a rare token contributes far more than a
match on a token that appears in half the pool.

The index is built per `(split, source, country)` and cached, so re-running retrieval
with different query parameters costs only retrieval time — this is what made the
budget sweeps in §9 affordable.

---

## 7. Rare-token selection

The query side is where the design differs from standard blocking. Instead of a fixed
blocking key (which either explodes on common tokens or misses on rare ones), each
query selects **only its rarest tokens** subject to a hard cost budget:

| parameter | value | meaning |
|---|---|---|
| `kq` | 100 | maximum tokens contributed per query |
| `dfmax` | 100000 | skip any token whose posting list exceeds this |
| `budget` | 80000 | cumulative posting-list length cap per query |

Tokens are sorted by document frequency ascending and consumed until the cumulative
posting-list length would exceed `budget`; tokens above `dfmax` are skipped as
uninformative. This makes per-query cost **bounded by construction and independent of
pool size** — the property that makes the pipeline scale, and the reason no
Source-1 × Source-2/Source-3 all-pairs comparison is ever performed.

It also adapts to how much evidence a record actually carries: a record with several
distinctive tokens contributes several, while a record of common words contributes
few and cheap ones.

---

## 8. Candidate generation

Retrieval returns the top `topk = 30` records from **each** target source per
Source-1 entity, giving **60 candidates per entity** (30 from Source 2 + 30 from
Source 3). Both pools are searched with the same query, so the two sources compete on
equal terms rather than one being preferred.

The candidate set is written out as `output/candidate_pairs.tsv` — the exact set the
final model scores, not a superset.

---

## 9. Candidate recall

Blocking quality is measured as **candidate recall**: the fraction of true pairs that
appear in the retrieved candidate list at all. Anything missed here is unrecoverable
downstream.

| query posting budget | 20,000 | 40,000 | 80,000 | 200,000 |
|---|---|---|---|---|
| candidate recall | 0.9368 | 0.9495 | **0.9552** | 0.9570 |

**0.9552 micro / 0.9547 macro** at the chosen budget of 80,000. The budget was tuned on
the validation split rather than guessed, and 80,000 is the knee of the curve: 200,000
costs 2.5× the runtime for +0.2pt. Down-weighting the address view (`w_addr = 0.5`) was
also tested and rejected — it costs 3.5pt of recall (0.9552 → 0.9200), confirming that
name and address evidence both carry signal.

Retrieval rank curve at the chosen budget: recall@3 = 0.8241, @5 = 0.8996,
@10 = 0.9299, @20 = 0.9473, @30 = 0.9552.

---

## 10. Pairwise features

45 features are computed per candidate pair (`src/features.py`):

- **Name similarity** — `ratio`, Jaro-Winkler (`nm_jw`), `partial_ratio`,
  `WRatio` on the normalized name; `token_set_ratio` (`nm_tset`) and
  `token_sort_ratio` (`nm_tsort`) on the canonicalized core; `ratio` on the
  sorted-unique-token form (`nm_sorted`, reorder-robust).
- **Address similarity** — the same family (`ad_ratio`, `ad_jw`, `ad_tset`,
  `ad_tsort`, `ad_partial`, `ad_sorted`, `ad_wratio`).
- **Numeric agreement** — `num_ratio` (distinct digit runs), `zip_ratio`, `zip_eq`,
  `zip_both`, `house_eq` — the ZIP / PIN / house-number evidence.
- **Rarity** — token document frequency from the retrieval index (`mindf_a`,
  `mindf_b`, `mindf_min`). See §18: the Source-1 side is currently zero-filled.
- **Retrieval context** — `score`, `rank_n` (rank within the candidate list),
  `score_ratio` (to the best candidate for that entity), `score_margin`, `ncand_n`.
  These let the model reason about *relative* evidence rather than absolute similarity.
- **Structural** — length and token-count ratios (`nm_len_ratio`, `nm_len_diff`,
  `ad_len_ratio`, `ad_len_diff`, `nm_tok_ratio`, `ad_tok_ratio`,
  `ntok_nm_a/b`, `ntok_ad_a/b`), script flags (`a_indic`, `b_indic`, `script_match`),
  the target-source indicator (`is_s3`), and empty-field indicators
  (`a_name_empty`, `b_name_empty`, `a_addr_empty`, `b_addr_empty`).

Top features by gain: `rank_n` 0.509, `house_eq` 0.128, `ad_tset` 0.040, `score`
0.038, `b_addr_empty` 0.037, `num_ratio` 0.029, `nm_partial` 0.025, `nm_ratio` 0.021,
`nm_wratio` 0.019, `nm_tset` 0.015.

The dominance of `rank_n` — where a candidate sits within its own list — is the
retrieval-then-rerank structure showing up in the model: relative position carries
more information than any single absolute string similarity.

Features are computed block-wise, one `(country, target-source)` block at a time, so
nothing is materialized for the whole split at once.

---

## 11. XGBoost model

| setting | value |
|---|---|
| library | XGBoost (Apache-2.0) |
| `tree_method` | `hist` |
| `max_depth` | 8 |
| `learning_rate` | 0.07 |
| early stopping | on a held-out tail of the training pairs |
| trees retained | **848** |
| blend | none (`blend_lr = 0.0`) |

A logistic regression on the same 45 features was trained as a baseline and scored
**0.8196** versus XGBoost's 0.9334. A 0.35/0.65 XGB+LR blend also under-performed at
**0.9217**. Both were rejected, so the tree model is used alone.

Raising the boosting cap to 3,000 rounds changed nothing (F0.5 0.9334 vs 0.9336),
because early stopping already selects 848 trees — evidence the model is at its
optimum for this feature set rather than truncated by the cap.

---

## 12. Class imbalance handling

Only ~5% of retrieved candidate pairs are true matches, and the raw ratio is worse
still. Two mechanisms are combined:

1. **Class-balanced negative downsampling** of the fit split.
2. **`scale_pos_weight = sqrt(spw)`** — the square root rather than the full inverse
   positive rate.

The square root is deliberate: the full weight over-corrects and pushes the model
toward recall, which is the wrong direction under F0.5. Partial correction keeps the
model calibrated enough that a single global threshold is meaningful.

Hard-negative mining was **not** used; class-balanced downsampling was chosen instead.

---

## 13. Threshold selection

The per-pair probability is thresholded to produce entity decisions. The threshold was
chosen by a **13-point sweep over 0.30–0.995** on the validation split, selecting the
maximum macro F0.5.

| threshold | F0.5 | precision | recall | avg predicted / S1 | empty |
|---|---|---|---|---|---|
| 0.80 | 0.9171 | 0.9364 | 0.9185 | 3.40 | 4.6% |
| 0.90 | 0.9297 | 0.9587 | 0.9020 | 3.26 | 5.4% |
| 0.93 | 0.9326 | 0.9671 | 0.8919 | 3.20 | 5.7% |
| **0.95** | **0.9334** | **0.9738** | **0.8811** | **3.14** | **6.0%** |
| 0.97 | 0.9322 | 0.9819 | 0.8627 | 3.05 | 6.5% |
| 0.99 | 0.9197 | 0.9927 | 0.8150 | 2.85 | 7.7% |

The optimum is **0.95**. The curve is flat on the precision side and steep on the
recall side above it, so the choice is stable rather than a knife-edge fit.

The ground-truth assignment constraint (a target id may back at most one Source-1
entity) was implemented and measured at **+0.0001** F0.5 — essentially neutral, because
candidate pairs are almost never contested — and was therefore dropped. See §18 for the
discrepancy this leaves in the saved config.

---

## 14. Singleton handling

No special rule is applied. Singletons are handled entirely by the threshold: if no
candidate for a Source-1 entity scores at or above threshold, the entity's match set
is written empty.

This is correct under the metric because empty truth plus empty prediction scores 1.0.
On validation the true singleton rate is 5.4% and the predicted-empty rate is 6.0%,
with singleton recall 0.879 and precision 0.781 — the model finds most true singletons
without forcing matches on entities that do have them. On the test split the
predicted-empty rate was 5.91%, against the 6.00% validation expectation.

An explicit "is this a singleton?" classifier was **not** built.

---

## 15. Validation

Validation is an **entity-level split on Source 1**: a fixed random sample of
`Source-1` entities is held out (`eval_blocking.make_split()`), with no pair leakage
between fit and validation. The **full** Source-2/Source-3 pools remain searchable, so
a validation entity competes against the same distractors it would face on test; the
split does not artificially shrink the pool.

| split | entities |
|---|---|
| fit | 60,000 Source-1 |
| validation | 30,000 Source-1 |

Split membership is cached (`cache/split.npz`) so every experiment is scored against
the identical held-out set.

Because the split is drawn from training data, it contains only US and India. It
therefore **cannot** measure French accuracy — see §18.

---

## 16. Error analysis

**False positives (wrong merges).** 2,482 predicted pairs are wrong — 2.64% of all
predictions, affecting 7.4% of Source-1 entities, almost always just one spurious pair
each (mean 1.12 per affected entity, max 5). They split evenly across pools (Source 2:
1,209, Source 3: 1,273). Their signature is a *plausible but weaker* match: against
true positives they show much lower name similarity (`nm_ratio` 0.627 vs 0.823), lower
address similarity (`ad_ratio` 0.620 vs 0.793), and weaker house-number agreement
(`house_eq` 0.375 vs 0.692) — yet nearly the same ZIP/PIN agreement (`zip_eq` 0.945 vs
0.963). These look like genuinely different businesses sharing a postcode, or
branch/subsidiary records that the ground truth does not treat as the same entity.

**False negatives (missed matches).** Of 104,079 true pairs, **7.41% are retrieved but
scored below threshold**, and **4.48% are never retrieved at all**. The model, not the
blocking stage, is the larger source of recall loss — the direct cost of a
precision-favouring threshold. Roughly 6,369 Source-1 entities lose at least one true
match this way.

`src/err_analysis.py` produces the aggregate figures; `src/errors.py` prints the actual
records behind the false positives and false negatives for qualitative inspection.

---

## 17. Scalability

- **Bounded per-query cost.** The posting-list budget caps the work for each query
  *before* any scoring happens, so runtime is predictable and independent of pool size.
- **No all-pairs comparison.** Each Source-1 entity is compared against at most 60
  records. On test that is 60 of ~5.2M records per country, a reduction ratio of
  0.99999398 — versus ~1.7 × 10¹³ evaluations for an exhaustive join.
- **Country-at-a-time processing.** Peak memory is bounded by one country's Source-1
  slice plus its two target pools, not the whole split.
- **Per-block caching.** Inference is split into `(country, target-source)` blocks, each
  cached independently, making the multi-hour test run resumable rather than
  all-or-nothing.
- **Small model.** 848 trees of depth 8 (~10⁵ nodes) is far below the 8-billion-parameter
  limit, and the whole pipeline is CPU-only.

An operational note that cost real time and is worth recording: the sparse product
`Q @ TT` can emit up to `QCHUNK × budget` non-zeros, so the default `QCHUNK=4000`
transiently allocates several GB. Two concurrent blocks inside one process then exceed
available RAM and the run begins **working-set thrashing** — measured at 6 GB of swap in
use, 23,531 page faults/s against ~0 MB/s of disk I/O, with progress output stopped for
hours while the process still burned two cores. It presents as a hang but is memory
pressure. `QCHUNK=1000 WORKERS=1` keeps a country's peak near 3 GB, and each block now
logs its rate and working set every 30 s so a stall is visible as falling throughput
rather than as silence.

---

## 18. Limitations

- **France is unmeasured.** The validation split contains no French entities, so no
  accuracy figure exists for the 259,452 French test entities (15% of the test set).
  France is processed by the same code path with no special-casing — verified by row
  count — but that is a correctness argument, not a measurement. **Not measured.**
- **Blocking discards 4.48% of true pairs**, which no downstream model can recover.
  The budget sweep shows the remaining loss is expensive to recover (200,000 costs 2.5×
  for +0.2pt).
- **Two of the 45 features are dead.** `mindf_a` and `mindf_min` are identically 0
  because the Source-1 record dict carries no `mindf` key and `features.decorate`
  zero-fills it; only `mindf_b` is live. Neither appears in the top 15 by gain, so the
  measured cost is nil, but it is a genuine defect rather than a modelling choice, and
  enabling it requires retraining.
- **The saved config declares an assignment constraint that inference does not apply.**
  `configs/best_config.json` records `"bij": true`, and the threshold sweep that
  selected 0.95 ran with the constraint applied, but `predict.py` does not implement it.
  Measured effect: +0.0001 F0.5, below the metric's reporting precision. It cannot be
  applied post-hoc without an unvalidated tie-break heuristic, because per-pair
  probabilities are not retained in the block caches.
- **Validation is 30,000 entities.** Differences below ~0.003 F0.5 were not treated as
  real when comparing configurations, so several "no change" results in the experiment
  log are equivalently "no detectable change at this sample size".
- **Hard-negative mining was not explored**, and no hyperparameter search beyond
  `max_depth` / `learning_rate` / boosting-rounds sanity checks was performed.

---

## 19. Reproduction

The challenge dataset is **not included** in this repository. Obtain authorized access
through the official challenge channels and place the files under `dataset/` (see the
README for the exact layout). Paths are resolved relative to the repository root by
`src/common.py`; no absolute paths are baked into the code.

```bash
python -m pip install -r requirements.txt

# 1. multi-view normalization of all six files → cached parquet   (~15 min)
python src/preprocess.py

# 2. fit the pairwise matcher + calibrate the threshold           (~13 min)
python src/train.py

# 3. test inference, one (country, source) block at a time        (hours; resumable)
QCHUNK=1000 WORKERS=1 python src/predict.py blocks

# 4. merge the cached blocks into the output TSVs
python src/predict.py assemble
```

Outputs land in `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The
trained model is written to `experiments/best/`.

**Verification.** `src/check_predict.py` replays the inference path over the 30,000
validation entities and recomputes macro F0.5, asserting it reproduces the
training-time number — it reports `F05=0.9334 P=0.9737 R=0.8811` against a
training-time reference of `F05=0.9334`, verdict `MATCH`. `src/verify_pairs.py` and
`src/smoke_align.py` are cheaper gates that check the stored feature matrix actually
corresponds to the stored pairs.

**Optional re-tuning.** `python src/tune_block.py` records the retrieval rank of every
true pair, giving candidate recall for any smaller budget; `python src/grid_blocking.py`
runs a cheap grid over `kq` / `budget` / `dfmax` / `topk`. Indices are cached, so each
configuration costs only retrieval time. `python src/eda.py` regenerates the dataset
and ground-truth summary; `python src/final_metrics.py` recomputes the reported metrics
from cached artifacts.

Expected wall-clock on 16 cores / 16 GB: preprocessing ~15 min, training ~13 min,
inference several hours (dominated by the India blocks).
