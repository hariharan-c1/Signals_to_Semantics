# Repository scope

The release combines a source-first research repository with a compact,
real-data demonstration. It is designed to be understandable on GitHub and
reproducible by a researcher who obtains the dataset and regenerates bulk
artifacts.

## Included

| Category | Included material |
|---|---|
| Core source | Detection, AV2 loading, actor tagging, and stable identifiers |
| Pipeline source | Selected S0 through S7 scripts |
| Configuration | Thresholds, feature schema, safe templates, and fixed split manifests |
| LLM material | Five prompt variants with sanitized illustrative identifiers |
| Contracts | Evidence and LLM-output JSON Schemas |
| Examples | Synthetic schema examples plus one real S0–S7 traceability case |
| Results | Final defense JSON and CSV summaries plus supporting integrity counts |
| Media | One compressed scenario clip and poster with a separate AV2 data notice |
| Figures | Regenerated architecture, evaluation-funnel, result-dashboard, and social-preview assets |
| Demonstration | Offline verifier for the real trace and executable synthetic S0/S1A signal detection |
| Project files | README, license, citation, CI, tests, dependency updates, and contributor guidance |

## Excluded

| Category | Reason |
|---|---|
| Raw Argoverse 2 data | Dataset license, size, and user-managed download |
| Bulk `results/` and `Repro/` trees | Generated artifacts, duplication, and repository size |
| Model checkpoints and serialized estimators | Large reproducible outputs, not source |
| PostgreSQL dumps | Generated data and possible local metadata |
| Bulk LLM response corpus and media | Generated data, cost, size, and provenance |
| Credentials and populated environment files | Security |
| Machine-specific absolute paths | Portability and privacy |
| Embedded `.git` directory | The archive did not contain a usable published history |
| Python caches | Generated files |
| Obsolete and duplicate script variants | One clear execution path is easier to audit |
| Baseline-adapted legacy modules | Redistribution terms were not established |
| Local database and human-validator repositories | These are separate supporting projects and are not part of this release |

## Publication rule

Generated experiment artifacts belong under `artifacts/` and are ignored by
Git. Add only final, documented summaries to `results/verified/` or a small
traceable case to `examples/`. Dataset-derived media requires a separate data
notice. Any new third-party-derived code must include a provenance review
before publication.
