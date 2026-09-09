# Third-party data and software notice

## Argoverse 2

This repository does not redistribute the Argoverse 2 Sensor dataset. Users
must obtain it separately and comply with the
[Argoverse terms of use](https://www.argoverse.org/about.html#terms-of-use).
Those terms identify Argoverse data and documentation as
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and
request attribution to:

> © 2021 Argo AI, LLC

The fixed split manifests in `configs/splits/` identify public AV2 logs.
Compact metrics in `results/verified/` and the real traceability case in
`examples/hero_scenario/` were produced from experiments on AV2. The case's
poster and clip have a separate notice in `examples/hero_scenario/media/`.
This repository's MIT license does not replace or override rights that apply
to AV2 or to dataset-derived materials.

The separate [Argoverse 2 API](https://github.com/argoverse/av2-api) is
MIT-licensed by its maintainers and is installed as an optional dependency.

## Baseline-derived work

The thesis archive's baseline-adapted modules were located under
`src/baseline_adapt/`. That directory is not included in this clean release,
and the retained pipeline does not import it.

If those modules are added later, retain upstream copyright and license
notices, identify the exact upstream commit, and document all modifications.

## Other dependencies

Python packages are not vendored. Each dependency remains under its own
license. See `pyproject.toml` for package names and version ranges.
