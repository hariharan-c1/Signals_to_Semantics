-- ============================================================
-- S7A — Build Views + Indexes (Scenario Retrieval Contract Layer)
-- Database: thesis_db (Postgres 16 + pgvector)
-- Scope: assumes S6 tables already exist and are populated
-- ============================================================

BEGIN;

-- ------------------------------------------------------------
-- 1) SAFETY CHECKS
-- ------------------------------------------------------------

-- Ensure pgvector exists
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

DO $$
DECLARE
  missing_tables TEXT := '';
  t TEXT;
  required TEXT[] := ARRAY[
    'scenario_windows',
    'window_scores',
    'scenario_actors',
    'actor_features',
    'gat_teacher_scores',
    'gat_inference_scores',
    'gt_window_truth',
    'llm_backends',
    'llm_predictions',
    's4_window_evidence',
    'scenario_embeddings_gemma'
  ];
BEGIN
  FOREACH t IN ARRAY required LOOP
    IF NOT EXISTS (
      SELECT 1
      FROM information_schema.tables
      WHERE table_schema='public' AND table_name=t
    ) THEN
      missing_tables := missing_tables || CASE WHEN missing_tables='' THEN '' ELSE ', ' END || t;
    END IF;
  END LOOP;

  IF missing_tables <> '' THEN
    RAISE EXCEPTION 'S7A abort: missing required tables in public schema: %', missing_tables;
  END IF;
END $$;

-- Embedding dimension consistency check for the embeddings table.
-- We enforce: scenario_embeddings_gemma.embed_dim must have exactly one distinct value.
DO $$
DECLARE
  n_dims INT;
BEGIN
  SELECT COUNT(DISTINCT embed_dim) INTO n_dims
  FROM public.scenario_embeddings_gemma;

  IF n_dims = 0 THEN
    RAISE NOTICE 'S7A: scenario_embeddings_gemma is empty (0 rows) — indexes/views can still be created.';
  ELSIF n_dims > 1 THEN
    RAISE EXCEPTION 'S7A abort: scenario_embeddings_gemma has multiple embed_dim values. Fix before indexing.';
  END IF;
END $$;

-- ------------------------------------------------------------
-- 2) INDEX CREATION
-- ------------------------------------------------------------
-- Notes:
-- - We index embeddings for fast ANN search (cosine distance).
-- - HNSW is great for interactive retrieval. (pgvector supports HNSW indexes.)
-- - If the installed pgvector build does not support HNSW, switch to IVFFlat.
-- - We also add btree indexes for common filters/joins.

-- btree indexes (safe even if already present)
CREATE INDEX IF NOT EXISTS idx_s7_windows_split_window
  ON public.scenario_windows (split_name, window_key);

CREATE INDEX IF NOT EXISTS idx_s7_llm_predictions_split_backend_window
  ON public.llm_predictions (split_name, backend_id, window_key);

CREATE INDEX IF NOT EXISTS idx_s7_llm_backends_split_prompt_backenddir
  ON public.llm_backends (split_name, prompt_type, backend_dir);

CREATE INDEX IF NOT EXISTS idx_s7_gt_split_window
  ON public.gt_window_truth (split_name, window_key);

CREATE INDEX IF NOT EXISTS idx_s7_gat_infer_split_window_rank
  ON public.gat_inference_scores (split_name, window_key, rank);

CREATE INDEX IF NOT EXISTS idx_s7_teacher_split_window_track
  ON public.gat_teacher_scores (split_name, window_key, track_uuid);

CREATE INDEX IF NOT EXISTS idx_s7_evidence_split_window
  ON public.s4_window_evidence (split_name, window_key);

CREATE INDEX IF NOT EXISTS idx_s7_embed_split_backend_window
  ON public.scenario_embeddings_gemma (split_name, backend_id, window_key);

-- pgvector ANN index (HNSW cosine)
-- IMPORTANT: For cosine, use vector_cosine_ops.
-- Tune ef_construction / m if needed later (we can do that in S7B profiling).
CREATE INDEX IF NOT EXISTS idx_s7_embed_hnsw_cosine
  ON public.scenario_embeddings_gemma
  USING hnsw (embedding vector_cosine_ops);

-- ------------------------------------------------------------
-- 3) CANONICAL VIEW CREATION
-- ------------------------------------------------------------
-- v_scenario_trace: one row per (split_name, backend_id, window_key)
-- It joins: windows + S1 score_final + evidence + GT + LLM + backend metadata
-- plus top-3 GAT inference scores aggregated into a compact json array.
--
-- This is the "contract view" that S7B and the UI can rely on.

CREATE OR REPLACE VIEW public.v_scenario_trace AS
WITH gat_top3 AS (
  SELECT
    split_name,
    window_key,
    jsonb_agg(
      jsonb_build_object(
        'rank', rank,
        'track_uuid', track_uuid,
        'score', score
      )
      ORDER BY rank
    ) AS gat_top3_json
  FROM public.gat_inference_scores
  WHERE rank <= 3
  GROUP BY split_name, window_key
)
SELECT
  -- Keys / provenance
  w.split_name,
  w.log_id,
  w.window_key,
  w.t_start,
  w.t_end,
  w.dur_s,
  w.detector_name,

  -- Window scoring (S1): retain only the final score.
  ws.score_final,

  -- Evidence (S4)
  e.t_on,
  e.t_peak,
  e.t_off,
  e.episode_score_final,
  e.peak_decel_mps2,
  e.near_crosswalk,
  e.near_stopline,
  e.has_close_actor,
  e.primary_side,
  e.road_type_hint,
  e.min_dist_to_crosswalk_m,
  e.min_dist_to_stopline_m,
  e.num_actors_in_drivable,
  e.num_actors_off_drivable,
  e.num_vru_near_crosswalk,
  e.num_ped_near_crosswalk,
  e.actor1_track_uuid AS s4_actor1_track_uuid,
  e.actor2_track_uuid AS s4_actor2_track_uuid,
  e.actor3_track_uuid AS s4_actor3_track_uuid,
  e.actor1_s3_score   AS s4_actor1_s3_score,
  e.actor2_s3_score   AS s4_actor2_s3_score,
  e.actor3_s3_score   AS s4_actor3_s3_score,
  e.actor1_category   AS s4_actor1_category,
  e.actor2_category   AS s4_actor2_category,
  e.actor3_category   AS s4_actor3_category,

  -- GAT inference (top3 as json)
  g3.gat_top3_json,

  -- Ground truth (if available)
  gt.label_canonical AS gt_label_canonical,
  gt.label_raw       AS gt_label_raw,
  gt.has_guest       AS gt_has_guest,
  gt.guest_track_uuid AS gt_guest_track_uuid,

  -- LLM backend metadata
  b.backend_id,
  b.prompt_type,
  b.backend_dir,
  b.model_name,
  b.provider_hint,

  -- LLM prediction fields
  p.scenario_label_raw       AS llm_label_raw,
  p.scenario_label_canonical AS llm_label_canonical,
  p.confidence_score         AS llm_confidence,
  p.primary_actor_raw        AS llm_primary_actor_raw,
  p.primary_track_uuid       AS llm_primary_track_uuid,
  p.actor1_track_uuid        AS llm_actor1_track_uuid,
  p.actor2_track_uuid        AS llm_actor2_track_uuid,
  p.actor3_track_uuid        AS llm_actor3_track_uuid,
  p.actor_matrix_json        AS llm_actor_matrix_json,
  p.final_rationale          AS llm_final_rationale,
  p.parsed_result_json       AS llm_parsed_result_json,

  -- Embedding (Gemma) – used for retrieval
  se.embedding_model,
  se.embed_dim,
  se.embedding,
  se.text_used

FROM public.llm_predictions p
JOIN public.llm_backends b
  ON b.backend_id = p.backend_id
JOIN public.scenario_windows w
  ON w.split_name = p.split_name
 AND w.window_key = p.window_key
LEFT JOIN public.window_scores ws
  ON ws.split_name = w.split_name
 AND ws.window_key = w.window_key
LEFT JOIN public.s4_window_evidence e
  ON e.split_name = w.split_name
 AND e.window_key = w.window_key
LEFT JOIN gat_top3 g3
  ON g3.split_name = w.split_name
 AND g3.window_key = w.window_key
LEFT JOIN public.gt_window_truth gt
  ON gt.split_name = w.split_name
 AND gt.window_key = w.window_key
LEFT JOIN public.scenario_embeddings_gemma se
  ON se.split_name = p.split_name
 AND se.backend_id = p.backend_id
 AND se.window_key = p.window_key
;

-- ------------------------------------------------------------
-- 4) REPRODUCIBILITY TABLES (Retrieval runs + results)
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.s7_retrieval_runs (
  run_id           BIGSERIAL PRIMARY KEY,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

  split_name       TEXT NOT NULL,

  query_type       TEXT NOT NULL,   -- text/by_example/constraints/disagreement
  query_text       TEXT,            -- for text query
  seed_window_key  TEXT,            -- for by_example query

  -- backend scope: single backend or a set (store as json for flexibility)
  backend_id       BIGINT,          -- optional
  backend_scope    TEXT,            -- e.g., 'all_12', 'prompt:CoT', 'model:gpt-5-mini'
  backend_ids_json JSONB,           -- e.g., [1,2,3,...]

  embedding_table  TEXT NOT NULL DEFAULT 'scenario_embeddings_gemma',
  embedding_model  TEXT NOT NULL DEFAULT 'embeddinggemma:latest',

  filters_json     JSONB NOT NULL DEFAULT '{}'::jsonb,
  top_k            INT NOT NULL DEFAULT 10,

  notes            TEXT
);

CREATE TABLE IF NOT EXISTS public.s7_retrieval_results (
  run_id            BIGINT NOT NULL REFERENCES public.s7_retrieval_runs(run_id) ON DELETE CASCADE,
  rank              INT NOT NULL,
  window_key        TEXT NOT NULL,
  backend_id        BIGINT, -- which backend produced this result row (important in multi-backend runs)
  similarity_score  DOUBLE PRECISION,
  rerank_score      DOUBLE PRECISION,
  score_breakdown_json JSONB,
  notes             TEXT,

  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT s7_retrieval_results_pk PRIMARY KEY (run_id, rank),
  CONSTRAINT s7_retrieval_results_run_window_uq UNIQUE (run_id, window_key)
);

CREATE INDEX IF NOT EXISTS idx_s7_results_run
  ON public.s7_retrieval_results (run_id);

CREATE INDEX IF NOT EXISTS idx_s7_results_window
  ON public.s7_retrieval_results (window_key);

COMMIT;

-- ============================================================
-- SANITY PROBES (run manually; these are comments on purpose)
-- ============================================================

-- 1) View rowcount should be ~ llm_predictions rows (671 for val50)
-- SELECT split_name, COUNT(*) AS n FROM public.v_scenario_trace GROUP BY 1 ORDER BY 1;

-- 2) Coverage: how many rows have embeddings attached?
-- SELECT split_name,
--        COUNT(*) AS n_rows,
--        SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) AS n_with_embedding
-- FROM public.v_scenario_trace
-- GROUP BY 1;

-- 3) Spot-check one log_id across all 12 configs:
-- SELECT log_id, window_key, prompt_type, backend_dir, llm_label_canonical, llm_confidence
-- FROM public.v_scenario_trace
-- WHERE split_name='val50' AND log_id='87ce1d90-ca77-363b-a885-ec0ef6783847'
-- ORDER BY prompt_type, backend_dir;

-- 4) ANN query test (by-example): “find nearest neighbors to a given (backend_id, window_key)”
-- Replace BACKEND_ID and WINDOW_KEY.
-- WITH q AS (
--   SELECT embedding
--   FROM public.scenario_embeddings_gemma
--   WHERE split_name='val50' AND backend_id=BACKEND_ID AND window_key='WINDOW_KEY'
-- )
-- SELECT
--   se.window_key,
--   1 - (se.embedding <=> (SELECT embedding FROM q)) AS cosine_sim
-- FROM public.scenario_embeddings_gemma se
-- WHERE se.split_name='val50' AND se.backend_id=BACKEND_ID
-- ORDER BY se.embedding <=> (SELECT embedding FROM q)
-- LIMIT 10;

-- 5) Confirm index exists:
-- SELECT indexname, indexdef
-- FROM pg_indexes
-- WHERE schemaname='public' AND indexname='idx_s7_embed_hnsw_cosine';
