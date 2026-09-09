# Verified result summaries

This directory contains the final quantitative results shown in the Master's
thesis defense presentation. The presentation is the authoritative source for
the headline values. Raw data and model checkpoints are not included.

| File | Purpose |
|---|---|
| `final_defense_metrics.json` | Machine-readable source of truth and evaluation contracts |
| `s1_s3_val50.json` | Final event-localization and held-out actor-ranking results |
| `s3_train650_oof_by_class.csv` | Development-scale out-of-fold actor-ranking results |
| `s5_prompt_ablation_val50.csv` | Final four-prompt comparison on the selected backend |
| `s5_class_performance_val50.csv` | Final class-wise semantic F1 values |
| `s7_strict_gt_top10_val50.csv` | Final class-wise strict GT-only Top-10 retrieval results |
| `baseline_comparison.csv` | Final comparison with *Why Braking?* |
| `stage_precision_recall_proxies.csv` | Cross-stage operating points from the thesis appendix |
| `s6_integrity_val50.json` | Supporting cross-stage referential-integrity counts |

The old 26-window S5 backend scoreboard and four-label aggregate retrieval
table were removed because they used different intermediate evaluation
contracts from the final defense. Read `docs/RESULTS.md` before comparing or
quoting metrics across stages.

The fixed split manifests identify Argoverse 2 logs, and these aggregate
metrics were derived from AV2 experiments. See `NOTICE.md` for attribution and
the applicable data terms.
