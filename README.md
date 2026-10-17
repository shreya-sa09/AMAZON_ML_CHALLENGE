# Amazon ML Challenge 2026: Business Entity Resolution

Given business records from three sources, find every Source 2 / Source 3 record that refers to the
same real-world business as each Source 1 entity. Scoring is macro F0.5 per Source 1 entity. The
full task description is in [`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md).

**Status:** complete pipeline, submission files generated and validated.
**Validation macro F0.5: 0.978** (precision 0.989, recall 0.955), measured end-to-end on held-out
Source 1 entities against all 10.3M training S2/S3 records.

---

## Repository layout

| path | what it is |
|---|---|
| `business_entity_resolution/` | **current solution**: runnable pipeline (`src/`), `README.md` with run steps, pinned `requirements.txt` |
| `Documentation_template.md` | methodology write-up for the final submission, filled in |
| `make_submission.py` | builds `<team>_submission.zip` in the required layout |
| `utils/validate_submission.py` | official format validator |
| `entity_resolution_solution/` | earlier prototype, kept for history (superseded, see below) |
| `PROBLEM_STATEMENT.md` | challenge statement (was the old root README) |
| `dataset/`, `cache/`, `output/` | data, intermediate caches, submission files. Git-ignored, never committed |

---

## Where the project started (earlier prototype)

The first commit had a research prototype in `entity_resolution_solution/`:

* data inspection, a stratified validation split, 4-pass exact-key blocking (full name, clean name,
  first two tokens, name prefix + house number), 16 string-similarity features, and
  LogisticRegression / RandomForest / XGBoost models;
* reported **F0.5 0.850**, but validated on a *sampled* corpus (true matches + 200k random records),
  which makes precision look better than on the full data;
* blocking recall was only **~70%**, which capped the achievable score;
* a follow-up "hybrid blocking" report (0.89–0.92) scored its models on the same data they were
  trained on, so those numbers are over-optimistic;
* paths were hard-coded to `c:\amazon_ml_challenge\...`, there was no test inference, no output
  files, no handling of France (test-only country) or of the Indic-script names, and the
  documentation template was empty.

## What changed

A new pipeline was written from scratch in `business_entity_resolution/`.

### 1. Data findings that drove the design
* Every matched S2/S3 record belongs to **exactly one** S1 entity and matches never cross countries,
  so the task is an assignment: each S2/S3 record goes to its best S1 entity, or to none.
* 26% of S2/S3 records match nothing. Many are near-copies of a real business with one word or one
  house number changed (hard negatives).
* About 7% of S2/S3 names are written in **9 Indic scripts** (Devanagari, Bengali, Gurmukhi,
  Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam).
* Other noise: website/hashtag names (`darkbuildwellclinic.com`), `doing business as`, digit-for-letter
  typos (`Preparat0ry`), shuffled word order, legal-suffix changes, `<NULL>`/`N/A` addresses, and
  state names vs. codes.
* Test adds **France** (15% of test S1), which is absent from training.

### 2. Pipeline

| stage | file | what it does |
|---|---|---|
| 0 | `prepare_data.py` | TSV → parquet |
| 1 | `normalization.py`, `build_normalized.py` | name/address normalization: one Indic→Latin transliteration table for all 9 scripts, accent folding, legal suffixes (US/India/France), `dba`, website names, leet-typo repair, state codes, street abbreviations (EN/FR) |
| 1b | `indic_dictionary.py`, `apply_indic_dict.py` | learns 526 transliteration→English token mappings (`praivet→private`, `injiniyaring→engineering`) from training pairs (training fold only) |
| 2 | `retrieval.py`, `candidates.py` | per country: TF-IDF of name character 3-grams (spaces removed) + address tokens; GPU CountSketch top-40 then exact re-score, keep **10 S1 candidates per S2/S3 record** |
| 3 | `features.py`, `build_features.py` | **64 features**: rapidfuzz name/address similarities, token and house-number overlap, "what differs" features (tokens/numbers on one side only), margins versus the runner-up candidate, ambiguity counts |
| 4 | `train.py`, `metrics.py` | LightGBM classifier (892 trees); validation that mirrors the leaderboard; threshold sweep |
| 5 | `predict.py` | assigns each record to its best candidate if p ≥ 0.7, then writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` |

### 3. Validation protocol
S1 entities are split 80/20 by id. Each S2/S3 record follows its true S1 entity (or its top
candidate if unmatched). The model and the Indic dictionary are fit on the 80% fold only. All 10.3M
records are then scored and macro F0.5 is computed on the 20% fold exactly as the leaderboard does.
Singletons are included and every distractor is present.

### 4. Results

| | candidate recall | macro F0.5 | precision | recall | India | US |
|---|---|---|---|---|---|---|
| earlier prototype (sampled corpus) | ~70% | 0.850 | 0.985 | 0.721 | — | — |
| new pipeline v1 | 98.1% | 0.9707 | 0.9835 | 0.9436 | 0.9608 | 0.9774 |
| **v2: + Indic dictionary + difference/margin features** | **98.7%** | **0.9783** | **0.9886** | **0.9549** | **0.9754** | **0.9802** |

* The Indic dictionary raised candidate recall for Indic-script records from 91.3% to 99.4%.
* Test has more distractors than train (about 40% vs 26%). Simulating that on validation gives about
  0.976, and the 0.7 threshold stays optimal.
* Test outputs: 1,732,544 S1 rows, 5.86M matched records, 99.7M candidate pairs.
  `utils/validate_submission.py` reports **PASS**.

### 5. Engineering notes
* Built for a 16 GB RAM laptop with an RTX 4050 (6 GB). Stages process one country or chunk at a
  time, cache to `cache/`, and resume after interruption.
* Uses only the provided data and open-source libraries (no external lookups, APIs or pretrained
  language models).

---

## How to run

See [`business_entity_resolution/README.md`](business_entity_resolution/README.md). In short:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install -r business_entity_resolution/requirements.txt
# place the challenge data in dataset/train and dataset/test
cd business_entity_resolution/src && ../../.venv/Scripts/python run_all.py
cd ../.. && .venv/Scripts/python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
.venv/Scripts/python make_submission.py <team_name>
```

The full run takes about 4–5 hours on the hardware above. Candidate generation and feature
computation take most of that time.

## Team Gamma:
Shreya Salunkhe,
Piu Maji,
Sakshi Harlekar.
