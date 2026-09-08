# Verified result summaries

This directory contains compact evaluation outputs selected from the thesis
experiment archive. Raw data, model checkpoints, per-window evidence, and LLM
responses are not included.

| File | Purpose |
|---|---|
| `s1_s3_val50.json` | Event-discovery and actor-ranking summary |
| `s6_integrity_val50.json` | Cross-stage trace and referential-integrity counts |
| `s6_backend_scoreboard_val50.csv` | Full S5 prompt and backend evaluation table |
| `s7_macro_micro_val50.csv` | S7 retrieval metrics by K |

Metrics have different denominators and evaluation contracts. Read
`docs/RESULTS.md` before comparing or quoting them.

The fixed split manifests identify Argoverse 2 logs, and these aggregate
metrics were derived from AV2 experiments. See `NOTICE.md` for attribution and
the applicable data terms.
