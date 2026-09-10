# Contributing

Contributions that improve reproducibility, tests, documentation, or data
contracts are welcome.

1. Create a focused branch.
2. Keep raw AV2 data, credentials, generated artifacts, and model weights out
   of Git.
3. Preserve upstream notices when adapting third-party code.
4. Run `make check` before opening a pull request.
5. Explain any change to an evaluation contract or reported metric.

The local publication gate matches CI:

```bash
python -m pip install -e .
make check
```

Changes to generated figures require the visualization extras:

```bash
python -m pip install -e ".[viz]"
make figures
make check
```

The figure manifest binds the generated files to the verified input ledgers.
Metric changes must also follow
[`docs/METRIC_PROVENANCE.md`](docs/METRIC_PROVENANCE.md).

Submitted code contributions may be distributed under this repository's MIT
License.
