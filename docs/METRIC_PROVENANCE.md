# Metric provenance

This document defines the public reporting policy for the final thesis
results. It exists because research artifacts can contain values from
different experiment revisions, aggregation rules, or draft narrative text.

## Authoritative values

| Metric | Public value | Derivation |
|---|---:|---|
| S3 Hit@1 | **23/26 = 88.5%** | Count of GT-aligned windows whose annotated responsible actor appears at Rank 1 |
| S3 Hit@3 | **26/26 = 100%** | Count of GT-aligned windows whose annotated responsible actor appears within the Top-3 |
| S5 accuracy | **0.760** | Base Prompt with the selected `gpt-5-chat` backend over 25 representative GT logs |
| S5 Macro-F1 | **0.621** | Macro average for the same selected S5 evaluation |

These values are encoded in
[`final_defense_metrics.json`](../results/verified/final_defense_metrics.json)
and used by the README, result tables, generated figures, tests, and demo.

## Resolution policy

The final defense results and their count-backed evaluation records are the
source of truth for this public repository. When draft thesis narrative,
historical experiment output, and the final count-backed record differ, the
following order is used:

1. Final count-backed evaluation record
2. Final defense result selected for public reporting
3. Historical or intermediate experiment output
4. Draft narrative text

Accordingly:

- S3 Hit@1 is reported as 23/26, or 88.5%. A legacy 0.893 scalar is not used
  as Hit@1 because it does not equal the final rank count.
- S5 Macro-F1 is reported as 0.621 for the selected Base Prompt configuration.
  Earlier rounded or draft headline values are not used.

The S3 value 0.893 may still appear in the thesis comparison material as a
separately defined stage-level precision proxy. It must not be relabelled as
Hit@1. Any future use of that proxy must state its evaluation contract
explicitly.

## Reproducible figures

Repository figures are regenerated from the committed verified ledgers:

```bash
python -m pip install -e ".[viz]"
make figures
python scripts/maintenance/generate_readme_figures.py --check
```

[`assets/figures_manifest.json`](../assets/figures_manifest.json) records the
SHA-256 digest of every input and output. CI checks the manifest without
requiring Matplotlib.

## Change control

A metric change must update all of the following in one pull request:

- the machine-readable ledger under `results/verified/`;
- the affected documentation and generated figures;
- the figure manifest;
- tests that encode the evaluation contract; and
- the pull-request explanation, including the reason for the change.

This prevents an isolated README edit from silently changing a research
claim.
