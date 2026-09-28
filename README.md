# Amazon ML Challenge 2026 — Business Entity Resolution

Entity resolution across three noisy business-record sources, using budget-bounded
sparse retrieval followed by a gradient-boosted pairwise matcher.

The pipeline reaches **0.9334 macro F0.5** on a held-out validation split and
**0.919 F0.5** on the final leaderboard. It uses only the challenge dataset — no
external business registry, geocoder, search engine, or pretrained entity embedding
is consulted at any point.

---

## Overview

The challenge provides three sources of business records:

- **Source 1** — the reference entity set. Each record is a business entity that must
  be resolved.
- **Source 2** and **Source 3** — noisy record pools containing the same underlying
  businesses, written differently: inconsistent punctuation, legal suffixes
  (`Pvt Ltd`, `LLC`, `SARL`), abbreviations, reordered tokens, partial strings, and
  mixed free-text addresses.

The task is to identify, for every Source-1 entity, **all** Source-2 and Source-3
records that refer to the same real-world business — or correctly determine that none
do. Records are joined by a shared `country` field that is treated as an open-set
string, so countries absent from training are handled by the same code path.

Scoring is **macro-averaged F0.5** (β = 0.5) computed per Source-1 entity, which
weights precision more heavily than recall. Predicting an empty match set for a true
singleton is scored as correct and carries full credit, so matches are never forced.

**Scale.** The test split contains 1,732,544 Source-1 entities and a 9,969,589-record
search pool across three countries. Comparing every entity against every record would
require ~1.7 × 10¹³ pair evaluations; the blocking stage reduces this to 60 candidates
per entity — a reduction ratio of 0.99999398.

---

## Results

### Validation

Measured on a held-out, entity-level split of **30,000 Source-1 entities** drawn from
the training data. The split is entity-level on Source 1 (no pair leakage), and the
full Source-2/Source-3 pools stay searchable, so candidate lists contain the same kind
of distractors as the test set.

| metric | value |
|---|---|
| **F0.5 (macro, per Source-1 entity)** | **0.9334** |
| Precision (micro, over pairs) | 0.9738 |
| Recall (micro, over pairs) | 0.8811 |
| Candidate recall (micro / macro) | 0.9552 / 0.9547 |
| Candidates per Source-1 entity (avg / median / p95 / max) | 60 / 60 / 60 / 60 |
| Entities receiving no candidate | 0.00% |
| Singleton recall / precision | 0.879 / 0.781 |
| Candidate reduction ratio | 0.99998837 |

### Final Leaderboard

| metric | value |
|---|---|
| **F0.5** | **0.919** |

**These numbers are not interchangeable.** 0.9334 is the validation score on a 30,000-entity
sample of US and India training data. 0.919 is the hidden-test score on 1,732,544
entities that additionally include **France**, a country that does not appear in the
training data at all. The gap is expected and is discussed under
[Limitations](#limitations).

---

## Pipeline

```
Raw business records (Source 1 / 2 / 3)
        │
        ▼
Multi-view Normalization
  unicode folding · punctuation · legal-suffix stripping
  abbreviation canonicalization · sorted-token · numeric views
        │
        ▼
Country-aware Blocking
  country as an opaque string — one hard block per country
        │
        ▼
Adaptive TF-IDF Retrieval
  inverted index of L2-normalized IDF weights
  each query keeps only its RAREST tokens under a posting-list budget
        │
        ▼
Candidate Generation
  top-k per target source → 60 candidates per Source-1 entity
        │
        ▼
Pairwise Feature Engineering
  45 features: string similarity · numeric/ZIP agreement
  token rarity · retrieval context · structural
        │
        ▼
XGBoost
  hist · depth 8 · lr 0.07 · early stopping (848 trees)
        │
        ▼
Threshold Optimization
  sweep over 0.30–0.995 → 0.95 maximizes macro F0.5
        │
        ▼
Entity Matching
  per-pair probability ≥ threshold → matched_entity_ids
```

The decisive property of the blocking stage is that the posting-list cost is capped
**before** any scoring happens, which makes per-query runtime predictable and
independent of pool size. No Source-1 × Source-2/Source-3 all-pairs comparison is ever
performed.

---

## Methodology

Full write-up: **[docs/methodology.md](docs/methodology.md)** — normalization,
blocking design, feature set, model and threshold selection, validation strategy,
error analysis, scalability, and limitations.

Results and metrics: **[results/results.md](results/results.md)**.
Final configuration: **[configs/best_config.json](configs/best_config.json)**.

---

## Repository layout

```
amazon-ml-entity-resolution/
├── README.md
├── requirements.txt
├── .gitignore
├── src/                      pipeline source (17 modules)
├── configs/
│   └── best_config.json      final blocking + model + threshold configuration
├── docs/
│   └── methodology.md        full technical write-up
└── results/
    └── results.md            measured metrics
```

### Source modules

| file | role |
|---|---|
| `common.py` | paths, TSV IO, normalization primitives, F0.5 metric |
| `preprocess.py` | multi-view normalization → cached parquet |
| `blocking.py` | inverted index, adaptive query construction, retrieval |
| `features.py` | the 45 pairwise features |
| `pairs.py` | builds labelled candidate pairs for a set of Source-1 entities |
| `train.py` | fits LR/XGB, sweeps the decision threshold, saves the best config |
| `predict.py` | test inference (block runner + assembly) and output writing |
| `eval_blocking.py` | validation split, ground truth, blocking metrics |
| `eda.py` | dataset and ground-truth EDA |
| `tune_block.py` | blocking re-tuning; records the retrieval rank of every true pair |
| `grid_blocking.py` | parameter grid over the blocking query budget |
| `final_metrics.py` | final metrics from cached artifacts |
| `err_analysis.py` | quantitative error analysis on the validation split |
| `errors.py` | qualitative error analysis — the records behind FP/FN |
| `check_predict.py` | verifies the inference path reproduces validation F0.5 |
| `verify_pairs.py` | checks the stored feature matrix matches the stored pairs |
| `smoke_align.py` | cheap alignment gate before a full pair rebuild |

---

## Dataset

**The challenge dataset is not included or redistributed in this repository.** The
data belongs to the Amazon ML Challenge and is not public. Users must obtain the
authorized challenge data separately through the official challenge channels and place
it in the expected `dataset/` directory:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

All files are TSV with a header and **no quoting** — business names and addresses
contain raw `"` and `,` characters, so every reader must disable quote handling or
fields get mangled. Paths are resolved relative to the repository root by
`src/common.py`; no absolute paths are baked into the code.

---

## Reproduction

Install dependencies, then run the four stages from the repository root. Preprocessing
caches to `cache/`, so stages are restartable.

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

Outputs land in `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

**This cannot be run end-to-end without the dataset.** The commands above are exact,
but stage 1 fails immediately unless the authorized challenge data is present under
`dataset/`. Nothing in this repository downloads or fabricates it.

**`QCHUNK=1000 WORKERS=1` is not cosmetic.** The sparse product in
`blocking.retrieve` can emit up to `QCHUNK × budget` non-zeros, so the default
`QCHUNK=4000` transiently allocates several GB per chunk; two concurrent blocks inside
one process then exceed available RAM and the run begins swapping. We hit exactly that:
6 GB of swap in use, 23.5k page faults/s, and progress output that stopped for hours
while the process still burned two cores. The smaller chunk and one source at a time
keep a country's peak near 3 GB.

**Expected wall-clock on 16 cores / 16 GB:** preprocessing ~15 min, training ~13 min,
inference several hours (dominated by the India blocks). `predict.py blocks` caches
each `(country, source)` block under `cache/predict_blocks/`, so an interrupted run
resumes rather than restarting.

**Verification.** `src/check_predict.py` replays the inference path over the 30,000
validation entities and recomputes macro F0.5, asserting it reproduces the
training-time number — it reports `F05=0.9334 P=0.9737 R=0.8811` against a
training-time reference of `F05=0.9334`, verdict `MATCH`.

---

## Model and license

The only model is an **XGBoost** booster fitted on the challenge's own training
labels. XGBoost is **Apache-2.0** licensed. The final model is 848 trees of depth 8
(≈10⁵ nodes), far below the 8-billion-parameter limit. No pretrained weights are used,
and no external entity data is consulted.

The trained model file is **not** included in this repository — it is reproducible from
the source and the dataset via the commands above.

---

## Limitations

These are known, measured properties of the system rather than open questions.

- **The leaderboard gap is mostly the unseen country.** Validation covers US and India;
  the test split adds France, which never appears in training. France flows through the
  same code path with no special-casing — verified by row count, not by assertion — but
  no accuracy figure exists for it, because the validation split contains no French
  entities. The 0.9334 → 0.919 change should be read with that in mind.

- **Blocking discards 4.48% of true pairs.** Candidate recall is 0.9552, so a fraction
  of correct answers are never retrieved and cannot be recovered by any downstream
  model. A budget sweep (20k → 0.9368, 40k → 0.9495, 80k → 0.9552, 200k → 0.9570)
  shows the chosen 80,000 is the knee; the remaining loss is expensive to recover.

- **The model, not the blocking stage, is the larger source of recall loss.** Of
  104,079 true pairs on validation, 7.41% are retrieved but scored below threshold,
  versus 4.48% never retrieved. This is the direct cost of F0.5: the same threshold
  that buys 0.9738 precision gives up that recall.

- **Residual false positives are same-postcode businesses.** 2,482 predicted pairs are
  wrong (2.64% of predictions), almost always one spurious pair per affected entity.
  They have much lower name similarity (`nm_ratio` 0.627 vs 0.823) and house-number
  agreement (0.375 vs 0.692) than true positives, yet nearly identical ZIP/PIN
  agreement (0.945 vs 0.963) — consistent with genuinely different businesses sharing
  a postcode, or branch/subsidiary records the ground truth does not treat as the same
  entity.

- **Two of the 45 features are dead.** `mindf_a` and `mindf_min` are identically 0
  because the Source-1 record dict carries no `mindf` key and is zero-filled; only
  `mindf_b` is live. Neither appears in the top 15 by gain, so the measured cost is
  nil, but it is a genuine defect rather than a modelling choice, and enabling it
  requires retraining.

- **The saved config declares an assignment constraint that inference does not apply.**
  `configs/best_config.json` records `"bij": true`, and the threshold sweep that chose
  0.95 was run with the constraint (each target id may back at most one Source-1
  entity) applied. The inference path does not implement it. It was measured at
  **+0.0001** F0.5 — below the metric's reporting precision — which is why it was
  dropped, but it cannot be applied after the fact without an unvalidated tie-break,
  because per-pair probabilities are not retained in the block caches.

- **The validation split is 30,000 entities, not the full training set.** Metrics
  carry sampling noise at that size; differences below ~0.003 F0.5 were not treated
  as real when choosing between configurations.
