# Repository scope

The release is intentionally source-first. It is designed to be understandable
on GitHub and reproducible by a researcher who obtains the dataset and
regenerates artifacts.

## Included

| Category | Included material |
|---|---|
| Core source | Detection, AV2 loading, actor tagging, and stable identifiers |
| Pipeline source | Selected S0 through S7 scripts |
| Configuration | Thresholds, feature schema, safe templates, and fixed split manifests |
| LLM material | Five prompt variants with sanitized illustrative identifiers |
| Contracts | Evidence and LLM-output JSON Schemas |
| Examples | Fully synthetic evidence and model output |
| Results | Small verified JSON and CSV summaries |
| Project files | README, license, citation, CI, tests, and contributor guidance |

## Excluded

| Category | Reason |
|---|---|
| Raw Argoverse 2 data | Dataset license, size, and user-managed download |
| Bulk `results/` and `Repro/` trees | Generated artifacts, duplication, and repository size |
| Model checkpoints and serialized estimators | Large reproducible outputs, not source |
| PostgreSQL dumps | Generated data and possible local metadata |
| LLM response corpus and media | Generated data, cost, size, and provenance |
| Credentials and populated environment files | Security |
| Machine-specific absolute paths | Portability and privacy |
| Embedded `.git` directory | The archive did not contain a usable published history |
| Python caches | Generated files |
| Obsolete and duplicate script variants | One clear execution path is easier to audit |
| Baseline-adapted legacy modules | Redistribution terms were not established |
| Local database and human-validator repositories | These are separate supporting projects and are not part of this release |

## Publication rule

Generated experiment artifacts belong under `artifacts/` and are ignored by
Git. Add only compact, documented summaries to `results/verified/`. Any new
third-party-derived code must include a provenance review before publication.
