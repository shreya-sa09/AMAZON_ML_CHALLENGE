# Experiment 07 — Candidate recall decomposition

Branch: `experiment/07-candidate-recall-decomposition`. This experiment reads the existing training ground truth, saved top-10 candidate chunks, baseline best rows, and Approach 02 validation scores. It does not alter those inputs or run the full candidate pipeline.

## Definitions and scope

- **Top-10:** the true S1 appears in the saved `cache/train_cands/*.parquet` rows for a matched S2/S3 query.
- **Top-1:** the true S1 is the *baseline LightGBM* best S1 in `cache/train_best.parquet`. The retrieval cosine rank-1 count is reported separately.
- **Accepted:** correct baseline top-1 and `p_final >= 0.70`, where `p_final` is Approach 02 Stage-2 probability within its band and baseline `p1` outside it.
- Stage-2 validation scores exist only for fold B (`S1 id % 5 == 0`), so the complete Top-10 → Top-1 → acceptance funnel uses that fold. Full-training Top-10 and Top-1 counts are reported separately. No full-training Stage-2 recall is claimed.

## Full training counts

7,638,365 ground-truth links correspond to 7,638,365 matched queries. Ground truth contains 123,247 singleton S1, 119,157 S1 with exactly one link, and 1,964,417 multi-match S1.

| Group | True links | Top-10 | Top-10 recall | Baseline model Top-1 | Top-1 recall | Top-10 → Top-1 loss |
|---|---:|---:|---:|---:|---:|---:|
| All | 7,638,365 | 7,538,403 | 98.6913% | 7,477,824 | 97.8982% | 60,579 |
| US | 4,578,522 | 4,529,240 | 98.9236% | 4,487,945 | 98.0217% | 41,295 |
| India | 3,059,843 | 3,009,163 | 98.3437% | 2,989,879 | 97.7135% | 19,284 |
| Query address present | 7,301,347 | 7,272,692 | 99.6075% | 7,262,665 | 99.4702% | 10,027 |
| Query address missing | 337,018 | 265,711 | 78.8418% | 215,159 | 63.8420% | 50,552 |

The separate **retrieval cosine rank-1** count is 7,412,895 (97.0482%). Baseline LightGBM reranking recovers some pairs from cosine ranks 2–10.

## Comparable validation funnel at threshold 0.70

| Group | True links | Top-10 recall | Baseline Top-1 recall | Accepted recall | Top-10 → Top-1 loss | Top-1 → final loss |
|---|---:|---:|---:|---:|---:|---:|
| All fold B | 1,529,739 | 98.6973% | 97.8489% | 96.6028% | 12,978 | 19,062 |
| US | 916,433 | 98.9240% | 97.9636% | 96.6226% | 8,801 | 12,290 |
| India | 613,306 | 98.3586% | 97.6775% | 96.5733% | 4,177 | 6,772 |
| Address present | 1,462,172 | 99.6141% | 99.4634% | 98.5928% | 2,204 | 12,729 |
| Address missing | 67,567 | 78.8565% | 62.9109% | 53.5380% | 10,774 | 6,333 |

The all-fold-B counts are: **1,509,811** Top-10, **1,496,833** model Top-1, and **1,477,771** accepted. Of 19,062 correct Top-1 rows rejected at 0.70, **18,909** are in-band Stage-2 rejections and **153** are baseline rejections outside the band. At 0.50, 1,483,085 are accepted (96.9502%).

## Top-40 sample diagnostic

**Top-40 recall cannot be measured from existing caches.** The approximate top-40 lists were not saved.

`src/sample_top40.py` reruns the *existing* country index on a fixed-seed sample of 2,000 known saved Top-10 misses (1,000 US, 1,000 India). It uses the same 40-neighbor approximate search and the same ordinal exact ranking rule. Runtime was 45 seconds on CUDA.

| Known saved Top-10 misses | In rerun Top-40 | Rerun exact Top-10 | Lost in Top-40 → Top-10 |
|---|---:|---:|---:|
| 2,000 | 621 (31.05%) | 0 | 621 |
| US 1,000 | 333 | 0 | 333 |
| India 1,000 | 288 | 0 | 288 |

Of 1,437 sampled misses with a missing query address, 556 enter Top-40. Of 563 with an address present, 65 enter Top-40. These are **conditional sample results among known Top-10 misses**, not full-corpus Top-40 recall. They show both an approximate-search ceiling (1,379 absent from Top-40) and a Top-10 pruning loss (621 present in Top-40), without estimating population totals from the balanced sample.

## Observed failure patterns

Full corpus: **99,962** Top-10 misses, of which **71,307 (71.3%)** have a missing query address. India has 50,680 misses and US 49,282. The query `has_indic` flag is set for 3,218 misses.

A separate fixed-seed sample of 200 Top-10 misses was joined to the true S1's normalized fields. Flags overlap and describe observations rather than causes:

- 135/200 lack a query address; 147/200 have a true S1 core name duplicated elsewhere in the same country, and 119/200 share that core name with at least ten S1 records.
- 57/200 have normalized name character similarity below 60; examples include shortened names and heavily changed names with partial addresses.
- 24/200 have conflicting first house numbers; 8/200 have the Indic query flag; 4/200 differ in numeric name tokens. Only 1/200 has address token-set similarity below 60 when both addresses are present.

See `results/top10_miss_sample_details.csv` for the paired normalized fields and exact flags. We do not assign an unsupported exclusive category such as "name corruption" to every low-similarity pair.

## Decision

**Next experiment: C — targeted increase/rework of retrieval K for missing-address queries.** They are only 4.4% of matched queries yet account for 71.3% of Top-10 misses and 83.4% of the full-corpus Top-10 → model Top-1 loss. In the targeted diagnostic, 556/1,437 missing-address misses enter Top-40 but are pruned before Top-10. Test a bounded, address-aware K policy on validation, including its Stage-1 cost and precision, before considering a broader retrieval method. This next experiment is not implemented here.

## Reproduce

From the repository root, using the existing project Python environment:

```powershell
C:\Users\shrey\.venv\Scripts\python.exe approaches\07_CANDIDATE_RECALL_DECOMPOSITION\src\decompose.py
C:\Users\shrey\.venv\Scripts\python.exe approaches\07_CANDIDATE_RECALL_DECOMPOSITION\src\sample_top40.py
C:\Users\shrey\.venv\Scripts\python.exe approaches\07_CANDIDATE_RECALL_DECOMPOSITION\src\analyze_misses.py
```

Only compact files under this approach's `results/` are written. `summary.json`, `top40_sample_summary.json`, and `failure_sample_summary.json` hold the primary numbers; four CSVs hold fixed samples and their diagnostics.
