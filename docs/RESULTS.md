# Verified results

This page reports compact results recovered from the thesis experiment
artifacts. The machine-readable summaries and source CSV files are in
`results/verified/`.

## S1 event discovery

| Measure | Value |
|---|---:|
| Final predicted windows | 56 |
| Distinct predicted windows aligned with available ground truth | 26 |
| Predicted windows without a ground-truth overlap | 30 |
| Ground-truth event recall | 1.000 |

The 30 unmatched predictions are called **candidate discoveries**. They are
not counted as verified new scenarios without a separate review.

## S3 actor ranking

| Measure | Value |
|---|---:|
| Annotated evaluation instances | 28 |
| Distinct predicted windows represented | 26 |
| Hit@1 | 0.893 |
| Recall@3 | 1.000 |
| Average rank | 1.179 |
| Mean reciprocal rank | 0.935 |
| Pairwise ranking accuracy | 0.911 |

The instance count and distinct-window count are intentionally reported
separately because more than one annotation can map to the same predicted
window.

## S5 reasoning evaluation

The archived scoreboard covers 12 combinations of four prompt formats and
three backends on 26 ground-truth windows.

| Selection criterion | Prompt | Model label | Coverage | Canonical score |
|---|---|---|---:|---:|
| Highest GT accuracy | base prompt | gpt-5-chat | 1.000 | Accuracy 0.577 |
| Highest GT macro-F1 | chain of thought | ollama_gpt-oss | 1.000 | Macro-F1 0.601 |

These are two different configurations. They must not be presented as if one
configuration achieved both values. Model labels reflect the archived run and
do not uniquely identify a provider-side model snapshot.

See [s6_backend_scoreboard_val50.csv](../results/verified/s6_backend_scoreboard_val50.csv)
for all rows and the raw and canonical label metrics. Human-validation columns
are retained as archived outputs, but their separate validation interface is
not included in this main repository.

## S6 trace integrity

| Check | Value |
|---|---:|
| Evidence files present | 56 / 56 |
| LLM predictions ingested | 671 |
| Invalid LLM JSON files | 0 |
| Orphan window references | 0 |
| Non-null primary actors mapped | 530 / 530 |
| Scenario-actor feature rows | 1,326 |
| S2 embedding rows | 782 |
| GAT inference rows | 168 |
| Ground-truth rows | 26 |
| Evaluated backend configurations | 12 |

The integrity checks verify traceability and key consistency. They do not by
themselves measure semantic correctness.

## S7 retrieval

The published `val50` retrieval summary averages over four labels with
sufficient ground-truth support.

The source table is
[s7_macro_micro_val50.csv](../results/verified/s7_macro_micro_val50.csv).

| K | Macro GT precision | Macro GT coverage | Micro GT precision | Micro GT coverage |
|---:|---:|---:|---:|---:|
| 1 | 0.438 | 0.090 | 0.380 | 0.070 |
| 3 | 0.250 | 0.140 | 0.237 | 0.120 |
| 5 | 0.225 | 0.210 | 0.234 | 0.190 |
| 10 | 0.267 | 0.466 | 0.290 | 0.440 |

“Coverage” is the metric name used by the evaluation output and should not be
silently relabeled as standard recall without preserving its contract.

## Interpretation limits

- Results are specific to the fixed `val50` evaluation setup.
- No raw AV2 data or learned checkpoints are included in this repository.
- LLM service revisions can change results.
- The available ground truth does not exhaust all meaningful braking causes.
- Results do not establish formal causal responsibility or production safety.
