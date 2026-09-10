# Final thesis results

This page reproduces the final values presented in the Master's thesis
defense on 3 September 2026. The author and thesis supervisor confirmed these
as the authoritative results for this repository. The machine-readable ledger
is [`final_defense_metrics.json`](../results/verified/final_defense_metrics.json).
The [metric-provenance policy](METRIC_PROVENANCE.md) explains how final
count-backed values supersede inconsistent legacy scalars.

![Final held-out results dashboard](../assets/results_dashboard.png)

## Evaluation contracts

| Stage | Evaluation unit | Contract |
|---|---|---|
| S1 | 50 held-out logs and 26 GT-aligned windows | Event preservation and internal temporal consistency |
| S3 | 26 distinct GT-aligned windows | Rank position of the annotated responsible actor |
| S5 | 25 GT logs | One deterministic representative window per log |
| S7 | Class-specific queries over Val50 | Strict GT-only relevance at `K=10`; unlabelled windows are not true positives |

These denominators answer different questions. Combining them into one
overall pipeline accuracy would be misleading.

![Held-out Val50 evaluation funnel](../assets/evaluation_funnel.png)

## S1 event localization

| Measure | Val50 result |
|---|---:|
| Held-out logs | 50 |
| Detected windows | 56 |
| Scored and retained windows | 56 |
| Hard-rejected windows | 0 |
| GT-aligned windows | 26 |
| Unlabelled candidate discoveries | 30 |
| GT event recall | 1.000 |
| Median onset latency | 0.0 s |
| Median temporal IoU | 1.000 |
| Mean fragmentation | 1.000 |

All 26 GT-aligned interactions were preserved. The 30 unmatched windows are
candidate discoveries because the available annotations do not assign them a
ground-truth event. They require separate review before they can be called
valid or novel scenarios.

| Represented scenario type | GT windows | Recall |
|---|---:|---:|
| `cut_in` | 10 | 1.00 |
| `approach_stop` | 5 | 1.00 |
| `obj_crossing` | 6 | 1.00 |
| `lead_brake` | 4 | 1.00 |
| `ped_crossing` | 1 | 1.00 |

The development-scale Train650 check detected 688 windows, including 263
with a ground-truth overlap, and retained GT recall of 1.00.

## S3 responsible-actor ranking

| Placement | Count | Cumulative rate |
|---|---:|---:|
| Rank 1 | 23 / 26 | 88.5% |
| Within Top-2 | 24 / 26 | 92.3% |
| Within Top-3 | 26 / 26 | 100% |
| Miss beyond Top-3 | 0 / 26 | 0% |

All Rank-1 misses remained in the Top-3 shortlist passed to semantic
reasoning. Agreement with the physics teacher, reported as NDCG@3, was 0.874
on GT-aligned windows and 0.883 on unlabelled candidates. Teacher agreement is
supporting ranking evidence, not ground-truth accuracy.

The approved Hit@1 value is the count-derived 23/26 = 88.5%. It is distinct
from the separately defined 0.893 stage-level precision proxy retained in the
thesis comparison material.

Development-scale out-of-fold results are preserved in
[`s3_train650_oof_by_class.csv`](../results/verified/s3_train650_oof_by_class.csv).

## S5 semantic reasoning

The selected configuration used the Base Prompt with the `gpt-5-chat` model
label. Evaluation used one deterministic representative window from each of
25 ground-truth logs.

| Metric | Selected result |
|---|---:|
| Accuracy | 0.760 |
| Macro precision | 0.639 |
| Macro recall | 0.633 |
| Macro-F1 | 0.621 |

### Prompt ablation on the selected backend

| Prompt | Accuracy | Macro-F1 |
|---|---:|---:|
| Base Prompt | **0.760** | **0.621** |
| CoT | 0.560 | 0.484 |
| CP | 0.680 | 0.397 |
| ALLINONE | 0.520 | 0.263 |

Additional prompt complexity did not improve this bounded task on the
selected backend.

### Class-wise semantic F1

| Scenario type | F1 |
|---|---:|
| `lead_brake` | 0.89 |
| `obj_crossing` | 0.80 |
| `cut_in` | 0.71 |
| `approach_stop` | 0.33 |

The dominant confusion was `approach_stop` predicted as `cut_in` in three
cases. Infrastructure-driven braking remained the main semantic weakness.

## S7 semantic retrieval

The final retrieval evaluation used strict GT-only relevance at `K=10`.
Unlabelled windows were not counted as true positives, which makes the
reported retrieval quality conservative when the annotations are incomplete.

| Scenario type | N | P@10 | R@10 |
|---|---:|---:|---:|
| `cut_in` | 10 | 0.40 | 0.40 |
| `approach_stop` | 5 | 0.20 | 0.20 |
| `obj_crossing` | 6 | 0.42 | 0.63 |
| `lead_brake` | 4 | 0.28 | 0.69 |
| `ped_crossing` | 1 | 0.20 | 1.00 |

The [hero scenario](../examples/hero_scenario/README.md) demonstrates a text
query that retrieved the same canonical event at Rank 3 with cosine similarity
0.541343 and hybrid score 0.724211.

## Comparison with [*Why Braking?*](https://arxiv.org/abs/2507.15874)

### Actor prioritization and semantic classification

| Task | Metric | Baseline | Thesis |
|---|---|---:|---:|
| Actor prioritization | Precision | 0.37 | 0.893 |
| Actor prioritization | Recall | 0.86 | 1.00 |
| Semantic classification | Precision | 0.39 | 0.639 |
| Semantic classification | Recall | 0.78 | 0.633 |
| Semantic classification | F1 | 0.52 | 0.621 |

The learned actor-ranking extension increased precision while preserving full
recall at the reported operating point. Semantic precision and F1 improved,
while semantic recall decreased.

### Out-of-distribution retrieval

Values show baseline to thesis.

| Scenario | P@10 | R@50 |
|---|---:|---:|
| `approach_stop` | 0.30 to 0.30 | 0.31 to 0.62 |
| `lead_brake` | 0.40 to 0.45 | 0.50 to 0.86 |
| `ped_crossing` | 0.30 to 0.75 | 0.67 to 0.84 |

R@50 improved in all three reported OOD categories.

## Cross-stage operating points

| Stage | Precision proxy | Recall proxy |
|---|---:|---:|
| S1 | 0.464 | 1.000 |
| S3 | 0.893 | 1.000 |
| S5 | 0.639 | 0.633 |
| S7 | 0.480 | 0.800 |

These are stage-specific proxies from the thesis appendix, not a
threshold-swept precision-recall curve. They show the system-level pattern:
S1 preserves events, S3 improves selectivity, S5 is the main bottleneck, and
S7 recovers coverage with a precision trade-off.

## Supporting trace-integrity checks

The final database audit found 56/56 evidence files, 671 ingested LLM
predictions, zero invalid JSON files, zero orphan window references, and
530/530 mapped non-null primary-actor references. These checks verify
referential integrity, not semantic correctness.

## Interpretation limits

- Results apply to the fixed Argoverse 2 Val50 evaluation within urban U.S.
  driving data.
- Human click timestamps limit external interpretation of the S1 temporal
  boundary diagnostics.
- Quasi-temporal graphs provide responsibility hypotheses, not proven physical
  causality.
- Semantic boundaries and model revisions can change S5 outputs.
- Transfer to other regions, datasets, and sensor configurations remains
  unverified.
