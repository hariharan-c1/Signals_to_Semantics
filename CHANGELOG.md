# Changelog

This file records meaningful changes to the public research artifact. The
project follows [Semantic Versioning](https://semver.org/) for its packaged
interfaces; research claims remain governed by the evaluation contracts and
provenance records in `docs/`.

## [1.0.0] - 2026-09-11

First stable public thesis artifact.

### Included

- Curated S0–S7 source code, configuration templates, schemas, and fixed split
  manifests.
- Verified final thesis metrics with machine-readable provenance.
- One real held-out trace covering event localization, actor ranking,
  evidence construction, semantic reasoning, human review, and retrieval.
- Deterministic offline demos and a 22-test publication gate across Python
  3.10 and 3.12.
- Recruiter-focused architecture, results, methodology, reproducibility, and
  limitation documentation.

### Maintenance

- Restricted CI triggers to `main` and pull requests targeting `main`.
- Applied read-only workflow permissions and cancellation of superseded runs.
- Upgraded GitHub Actions to v7 and pinned them to immutable commit SHAs.
- Grouped future GitHub Actions updates and retained manual review of research
  dependencies to protect the recorded thesis environment.

### Scope

- The release is an offline research and scenario-validation artifact, not a
  production autonomous-driving system or a claim of formal causality.
- Raw Argoverse 2 data, model checkpoints, credentials, bulk experiment
  outputs, and baseline-adapted code are intentionally excluded.

[1.0.0]: https://github.com/hariharan-c1/Signals_to_Semantics/releases/tag/v1.0.0
