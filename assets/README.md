# Repository figures

These recruiter-facing figures are regenerated from the committed final-result
ledger rather than copied from screenshots. Run:

```bash
python -m pip install -e ".[viz]"
make figures
```

`figures_manifest.json` records the SHA-256 digest of every input and output.
The repository checks can therefore verify that a displayed chart still
corresponds to the approved data without installing Matplotlib:

```bash
python scripts/maintenance/generate_readme_figures.py --check
```

## Thesis lineage

| Repository figure | Thesis source |
|---|---|
| `architecture_overview` | Redesigned from Figure 4.1 |
| `evaluation_funnel` | Redesigned from Figure 6.1 |
| `results_dashboard` | Regenerated from Figure 6.6, Figure 6.11, and Table 6.11 |
| `social_preview` | Composed from the verified final metric ledger |

The figures use the approved count-derived S3 value of 23/26 (88.5%) and the
final S5 Macro-F1 value of 0.621. Resolution of historical or differently
defined scalars is documented in
[`docs/METRIC_PROVENANCE.md`](../docs/METRIC_PROVENANCE.md).
