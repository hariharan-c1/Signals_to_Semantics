# End-to-end pipeline

This document describes how information moves from Argoverse 2 motion signals
to a searchable scenario trace. Run commands from the repository root.

## Shared conventions

- Local dataset paths live in `configs/paths.yaml`, which is ignored by Git.
- Generated files live under `artifacts/`, which is also ignored.
- A logical window key is `log_id|t_start|t_end` with timestamps rounded to
  six decimal places.
- S4 uses filesystem-safe hashed names for evidence JSON files and records the
  logical window key inside each JSON and in its manifest.
- Fixed log lists are stored under `configs/splits/`.

## Stage contracts

| Stage | Required inputs | Principal outputs | Validation |
|---|---|---|---|
| S0 | AV2 Sensor logs, thresholds, tagging configuration | Candidate-window JSONL, tagged actors, actor-feature Parquet | `validate_features.py` |
| S1 | Window features and labels | PU-XGB scores, nnPU scores, HMM-smoothed scores, finalized episodes | Calibration and summary files produced by S1 scripts |
| S2 | S1-kept windows and actor features | Actor index, contrastive tensors, 128-dimensional embeddings | `s2_validate_actor_index.py` and nearest-neighbor recall |
| S3 | Actor index, embeddings, S1 episodes, labels | Slices, graph files, soft teacher targets, GAT top-k ranking | S3 validation and evaluation scripts |
| S4 | S1 final scores, S3 top-k, actor features | Evidence JSON files and a manifest | `s4_2_validate_llm_input.py` |
| S5 | S4 evidence and a prompt template | Per-window LLM output JSON | `s5_validate_outputs.py` and `s5_evaluate_reasoning.py` |
| S6 | S0 to S5 artifacts plus PostgreSQL | Normalized scenario trace, embeddings, integrity and evaluation reports | `s6i_sanity_checks.py` and `s6_eval_all.py` |
| S7 | S6 scenario trace and text embeddings | Ranked retrieval results and aggregate metrics | `s7c_evaluate_retrieval.py` |

## S0: candidate detection and actor features

Prepare the local data configuration:

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

Edit `configs/paths.yaml`, then verify that every committed log ID exists:

```bash
python scripts/s0/validate_splits.py \
  --splits_dir configs/splits \
  --train_root /absolute/path/to/argoverse2/sensor/train
```

Detect, tag, and featurize one split:

```bash
python scripts/s0/detect_braking_windows.py \
  --paths configs/paths.yaml \
  --log_list configs/splits/val50.txt \
  --thresh configs/thresholds.yaml \
  --out_dir artifacts/train650_val50/val50/detect \
  --concat_out artifacts/train650_val50/val50/detect/brakes_all_union.jsonl

python scripts/s0/tag_candidate_actors.py \
  --paths configs/paths.yaml \
  --tag configs/tagging.yaml \
  --in artifacts/train650_val50/val50/detect/brakes_all_union.jsonl \
  --out artifacts/train650_val50/val50/tag/val50_tagged.jsonl

python scripts/s0/build_actor_features.py \
  --windows-jsonl artifacts/train650_val50/val50/tag/val50_tagged.jsonl \
  --paths-yaml configs/paths.yaml \
  --out-parquet artifacts/train650_val50/val50/features/val50_features.parquet

python scripts/s0/validate_features.py \
  --parquet artifacts/train650_val50/val50/features/val50_features.parquet
```

Repeat this step for development and training manifests as needed. The feature
contract is documented in `configs/features/schema_v1.yaml`.

## S1: event verification

S1 uses two positive-unlabeled learners, followed by score fusion and HMM
smoothing.

Run the scripts in this order:

1. `s1a_make_window_labels.py`
2. `s1a_build_window_table.py`
3. `s1a_train_pu_xgb.py`
4. `s1a_score_only_pu_xgb.py`
5. `s1a_calibrate_tau_cv.py`
6. `s1a_apply_threshold.py`
7. `s1a_prepare_train_sets.py`
8. `s1b_train_nnpu.py`
9. `s1b_inference.py`
10. `s1c_hmm_smooth.py`
11. `s1d_finalize_scores.py`

The scripts are under `scripts/s1_verifier/`. Use `python <script> --help`
for the required paths. Keep development-set calibration separate from the
held-out validation run.

## S2: actor embeddings

Run:

1. `s2_build_actor_index.py` to select actors around kept windows.
2. `s2_validate_actor_index.py` to enforce index invariants.
3. `s2_pack_contrastive.py` to create train and validation tensors.
4. `s2_train_contrastive.py` to train the projection network.
5. `s2_infer_contrastive.py` to export actor embeddings for each split.
6. `s2_nearest_neighbor_recall.py` for the representation diagnostic.

All scripts are in `scripts/s2_actor_emb/`.

## S3: quasi-temporal graph actor ranking

S3 converts the actor timeline into graph examples and ranks actors:

1. Create weighted event slices with `s3a1_make_slices.py`.
2. Create phase-specific pseudo frames with `s3a2_make_pseudo_frames.py`.
3. Build and validate PyTorch Geometric graphs with
   `s3a3_build_graphs.py` and `s3a3_validate_graphs.py`.
4. Build and validate soft teacher targets with `s3a4_make_teacher.py` and
   `s3a4_validate_teacher.py`.
5. Train the GAT with `s3a5_train_gat.py`.
6. Export and validate top-k actors with `s3a5_infer_gat.py` and
   `s3a5_validate_topk.py`.
7. Run `s3a5_evaluate_gat.py` or `evaluate_s1_to_s3.py`.

The committed implementation supports explicit paths for held-out evaluation
instead of relying on machine-specific defaults.

## S4: evidence construction

S4 joins S1, S3, map, and actor-feature outputs.

```bash
python scripts/s4/s4_1_build_evidence.py \
  --slices <slices.parquet> \
  --final-scores <final_scores.parquet> \
  --top3 <top3_infer.parquet> \
  --features <actor_features.parquet> \
  --outdir artifacts/train650_val50/val50/s4 \
  --split val50 \
  --brakes <brakes_all_union.jsonl> \
  --window-labels <window_labels.jsonl>

python scripts/s4/s4_2_build_llm_input.py \
  --evidence-dir artifacts/train650_val50/val50/s4/val50

python scripts/s4/s4_2_validate_llm_input.py \
  --evidence-dir artifacts/train650_val50/val50/s4/val50
```

A fully synthetic example is in
`examples/evidence/synthetic_window.json`.

## S5: LLM reasoning

The default backend is local Ollama. First generate prompts without making a
network call:

```bash
python scripts/s5/s5_llm_reason.py \
  --config configs/llm_backends.yaml \
  --backend ollama-local \
  --prompt-file scripts/s5/prompts/base_prompt.txt \
  --single-file examples/evidence/synthetic_window.json \
  --output-dir artifacts/s5_synthetic \
  --dry-run-prompts-dir artifacts/s5_dry_run
```

Remove `--dry-run-prompts-dir` to call the selected backend. Azure credentials
are read only from environment variables. Validate outputs with:

```bash
python scripts/s5/s5_validate_outputs.py \
  --llm-dir artifacts/s5_synthetic \
  --out-csv artifacts/s5_synthetic/sanity.csv
```

The prompt variants under `scripts/s5/prompts/` include base, concise,
chain-of-thought, all-in-one, and local-model formats. Few-shot identifiers in
the public prompt files are synthetic.

## S6: scenario database

S6 uses PostgreSQL and the `pgvector` extension.

```bash
cp configs/db.env.example configs/db.env
```

Edit only the ignored `configs/db.env`. Review artifact paths in
`configs/s6_db.yaml`, then run `s6a` through `s6i` in filename order. Each
ingestion step is idempotent at its documented database key. Finish with:

```bash
python scripts/s6/s6i_sanity_checks.py --config configs/s6_db.yaml
python scripts/s6/s6_eval_all.py --config configs/s6_eval.yaml
```

## S7: semantic retrieval

Create the database views and indexes:

```bash
psql -f scripts/s7/s7a_build_views_and_indexes.sql
```

Run a natural-language query:

```bash
python scripts/s7/s7b_retrieve.py \
  --config configs/s7_db.yaml \
  --query_text "A pedestrian crosses ahead and ego brakes to yield" \
  --top_k 10 \
  --no_store_to_db
```

Evaluate the fixed query suite:

```bash
python scripts/s7/s7c_evaluate_retrieval.py --config configs/s7_eval.yaml
```

## Artifact discipline

Do not commit files produced by these commands. The source repository keeps
only compact verified summaries. Record the Git commit, configuration, random
seed, environment, model checkpoint checksum, and LLM deployment identifier
for each experiment.
