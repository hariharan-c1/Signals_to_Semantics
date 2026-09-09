# Provenance and attribution

## Authorship

All code retained in this clean S0 to S7 release was developed by Hariharan
Chandrasekaran as part of the Master's thesis implementation.

The only baseline-adapted code in the supplied source archive was located
under `src/baseline_adapt/`. That complete directory, including its adapters,
dataset wrappers, taggers, feature bridges, constants, semantic bins, and
smoothing utilities, is excluded from this release. No code from that
directory is imported by the published pipeline.

If third-party-derived code is added later:

1. identify the upstream repository and commit;
2. confirm that its license allows redistribution;
3. retain its copyright and license notices;
4. mark modified files and summarize the changes;
5. avoid claiming the adapted portions as wholly original.

## Dataset provenance

The pipeline targets the Argoverse 2 Sensor dataset. Raw data is not included.
The committed split manifests contain public log identifiers, and the compact
result tables were derived from experiments on those logs. The real
traceability case under `examples/hero_scenario/` contains derived JSON output,
a poster, and a compressed clip from one held-out log. Its media directory
contains a separate AV2 data notice.

Consult:

- [Argoverse terms of use](https://www.argoverse.org/about.html#terms-of-use)
- [Argoverse 2 API repository](https://github.com/argoverse/av2-api)
- [NOTICE.md](../NOTICE.md)

## External model services

S5 supports Azure OpenAI and local Ollama endpoints. The repository includes
prompt text, aggregate results, and one selected structured response, but no
provider credentials. The final evaluation calls the selected model
`gpt-5-chat`; the selected raw response retains the archived provider hint
`azure-gpt-5-mini`. These labels do not uniquely identify a provider-side
model snapshot.

## Clean-release transformation

The public tree was selected from the thesis archive. The following cleanup
was applied without modifying the original archive:

- retained one coherent S0 to S7 implementation path;
- removed embedded Git metadata, caches, credentials, local paths, checkpoints,
  database dumps, and bulk generated artifacts;
- excluded obsolete duplicates and known broken variants;
- replaced dataset-specific identifiers in illustrative few-shot prompts with
  synthetic identifiers;
- moved shared imports into the `signals_to_semantics` package;
- replaced credential literals with environment-variable references;
- added cross-platform evidence filenames, documentation, tests, schemas, and
  repository checks.
- added one compact end-to-end case selected from the final defense
  demonstration, with original-source checksums and a separate media notice.

This document records the release's curation and attribution boundary.
