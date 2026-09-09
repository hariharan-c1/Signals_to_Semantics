# Reproducibility guide

## What this release supports

This repository provides the code path, split manifests, thresholds, feature
schema, prompt variants, configuration templates, evaluation scripts, and
compact final summaries needed to inspect and rerun the thesis pipeline.

It does not include:

- raw Argoverse 2 data;
- bulk generated Parquet, JSONL, media, or database artifacts;
- trained model checkpoints;
- the full commercial LLM response corpus;
- credentials or machine-specific paths.

Those omissions keep the repository publishable and small, but a complete
rerun requires regenerating intermediate artifacts. One compact, real
traceability case is retained under `examples/hero_scenario/` so the published
stage contracts can be inspected without a full rerun.

## Reference protocol

| Item | Value |
|---|---|
| Dataset | Argoverse 2 Sensor train split, downloaded separately |
| Development manifest | `configs/splits/dev100.txt`, 100 logs |
| Training manifest | `configs/splits/train650.txt`, 650 logs |
| Held-out validation manifest | `configs/splits/val50.txt`, 50 logs |
| Split seed | 1337 |
| Reference Python | 3.10 |
| Core validation also checked on | 3.12 |
| Recorded NumPy | 1.26.4 |
| Recorded pandas | 2.2.3 |
| Recorded PyTorch | 2.5.1+cu121 |
| Recorded CUDA | 12.1 |
| Recorded AV2 API | 0.3.5 |
| S1 temporal model | HMM smoothing |
| Primary output root | `artifacts/train650_val50/` |

`train650` is `dev100 ∪ train550`. The held-out `val50` set is disjoint from
all three training-related manifests.

## Recreate the software environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[full,dev]"
python -m pip freeze > artifacts/environment.freeze.txt
```

The exact PyTorch package should match the host's CPU or CUDA runtime. Record
the resolved versions because PyTorch Geometric wheels and GPU kernels may
change across platforms.

## Verify the clean repository

These checks require no AV2 download:

```bash
python scripts/maintenance/check_repository.py
python -m unittest discover -s tests -v
```

After downloading AV2, verify the fixed manifests:

```bash
python scripts/s0/validate_splits.py \
  --splits_dir configs/splits \
  --train_root /absolute/path/to/argoverse2/sensor/train
```

## Run order

Follow [PIPELINE.md](PIPELINE.md) from S0 through S7. Treat every stage as a
data contract:

1. Run its validator immediately after generation.
2. Preserve the configuration beside the generated artifact.
3. Record row counts and unique-key counts.
4. Compute a checksum for learned checkpoints and final evaluation tables.
5. Do not tune on `val50`.

## Randomness

The split, most representation-learning scripts, and graph-learning scripts
use explicit seeds. Some GPU operations can still be nondeterministic. Record:

- seed values;
- CPU or GPU model;
- CUDA, PyTorch, and PyTorch Geometric versions;
- training checkpoint checksum;
- command-line overrides.

## LLM reproducibility

LLM behavior is not fully deterministic even with temperature zero. For each
run, retain:

- provider and endpoint family;
- exact model or deployment identifier;
- request date;
- prompt file checksum;
- inference parameters;
- raw response, parsed JSON, and parse status;
- evidence-packet checksum.

The public Azure configuration resolves credentials from environment
variables. The default local configuration targets Ollama so prompt generation
and local inference do not require an Azure secret.

## Database reproducibility

S6 and S7 require PostgreSQL with `pgvector`. Use a new database or schema for
a rerun, apply `scripts/s7/s7a_build_views_and_indexes.sql` only after S6
ingestion, and run `s6i_sanity_checks.py` before retrieval evaluation.

Database passwords belong only in the ignored `configs/db.env` file.

## Evaluation boundaries

Metrics in `results/verified/` reproduce the final defense presentation and
use different denominators and contracts. Do not combine them into a single
percentage. In particular:

- an unmatched S1 prediction is a candidate discovery, not automatically a
  confirmed novel scenario;
- S3 reports rank placement on 26 distinct GT-aligned windows;
- S5 uses one deterministic representative window from each of 25 GT logs;
- S7 reports class-wise strict GT-only retrieval at `K=10`;
- the S1, S3, S5, and S7 precision/recall values in the cross-stage figure are
  stage-specific proxies, not points on one threshold-swept curve.

The released evidence supports offline scenario mining and reasoning on the
documented split. It does not prove causal attribution, real-time safety, or
generalization to other datasets.
