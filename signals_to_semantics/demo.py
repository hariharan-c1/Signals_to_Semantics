"""Dependency-light demonstrations for the public research artifact.

The real demonstration validates the published, recorded S0-S7 hero trace. It
does not rerun trained models or call external services. The synthetic signal
demonstration executes the released braking detector on generated kinematics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from signals_to_semantics.detection import detect_brakes
from signals_to_semantics.identifiers import canonical_window_key


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HERO_CASE = REPOSITORY_ROOT / "examples" / "hero_scenario"


class DemoValidationError(ValueError):
    """Raised when a published demonstration artifact fails validation."""


@dataclass(frozen=True)
class HeroTraceSummary:
    """Recruiter-readable summary of the verified hero scenario."""

    case_id: str
    window_key: str
    duration_s: float
    minimum_acceleration_mps2: float
    minimum_jerk_mps3: float
    s1_hmm_posterior: float
    candidate_actors: int
    top_actor: str
    top_actor_score: float
    model_label: str
    model_confidence: float
    human_label: str
    human_primary_actor: str
    retrieval_rank: int
    retrieval_hybrid_score: float
    checksums_verified: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoValidationError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DemoValidationError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_hero_trace(case_dir: Path | str = DEFAULT_HERO_CASE) -> HeroTraceSummary:
    """Validate and summarize the recorded real S0-S7 trace.

    Validation covers file checksums, canonical identity, stage-to-stage actor
    continuity, the model-human label comparison, and the S7 reranking formula.
    """
    case_dir = Path(case_dir)
    manifest = _read_json(case_dir / "manifest.json")
    errors: list[str] = []

    published = manifest.get("published_files", [])
    _require(isinstance(published, list) and bool(published), "Manifest has no files", errors)
    verified = 0
    for artifact in published if isinstance(published, list) else []:
        relative = artifact.get("path")
        expected = artifact.get("sha256")
        path = case_dir / str(relative)
        if not path.is_file():
            errors.append(f"Missing published artifact: {relative}")
            continue
        actual = _sha256(path)
        if actual != expected:
            errors.append(f"Checksum mismatch: {relative}")
            continue
        verified += 1

    s0 = _read_json(case_dir / "s0_candidate_window.json")
    s1 = _read_json(case_dir / "s1_verification.json")
    s2 = _read_json(case_dir / "s2_representation_summary.json")
    s3 = _read_json(case_dir / "s3_actor_ranking.json")
    s4 = _read_json(case_dir / "s4_evidence_pack.json")
    s5 = _read_json(case_dir / "s5_llm_output.json")
    human = _read_json(case_dir / "human_validation.json")
    s7 = _read_json(case_dir / "s7_retrieval_trace.json")

    canonical = manifest.get("canonical_window", {})
    source_key = canonical.get("window_key")
    _require(bool(source_key), "Manifest has no canonical window key", errors)
    _require(s0.get("log_id") == canonical.get("log_id"), "S0 log identity differs", errors)
    for label, observed in {
        "S1": s1.get("window_key"),
        "S2": s2.get("window_key"),
        "S3": s3.get("window_key"),
        "S4": s4.get("window_key"),
        "S5": s5.get("parsed_result", {}).get("ego_window_key"),
        "human review": human.get("window_key"),
        "S7 source": s7.get("canonical_source_window_key"),
    }.items():
        _require(observed == source_key, f"{label} window identity differs", errors)

    _require(s1.get("retained") is True, "S1 did not retain the event", errors)
    actors = s3.get("actors", [])
    _require(len(actors) == 3, "S3 must publish exactly three ranked actors", errors)
    if len(actors) == 3:
        _require([a.get("rank") for a in actors] == [1, 2, 3], "S3 ranks are invalid", errors)
        _require(len({a.get("track_uuid") for a in actors}) == 3, "S3 actors are not unique", errors)

    parsed = s5.get("parsed_result", {})
    model_actor = parsed.get("primary_responsible_actor")
    actor_aliases = {a.get("alias") for a in actors}
    _require(model_actor in actor_aliases, "S5 selected an actor outside the S3 Top-3", errors)
    _require(
        human.get("primary_trigger") == model_actor,
        "Model and human review disagree on the primary actor",
        errors,
    )

    weights = s7.get("rerank_weights", {})
    expected_hybrid = (
        float(weights.get("cosine_similarity", 0.0)) * float(s7.get("cosine_similarity", 0.0))
        + float(weights.get("llm_confidence", 0.0)) * float(s7.get("llm_confidence", 0.0))
        + float(weights.get("s1_confidence", 0.0)) * float(s7.get("s1_confidence", 0.0))
    )
    _require(
        abs(expected_hybrid - float(s7.get("hybrid_score", -1.0))) < 1e-10,
        "S7 hybrid score does not match its documented formula",
        errors,
    )
    _require(
        1 <= int(s7.get("rank", 0)) <= int(s7.get("top_k", 0)),
        "S7 rank lies outside the retrieval budget",
        errors,
    )

    if errors:
        raise DemoValidationError("Hero trace validation failed:\n- " + "\n- ".join(errors))

    top_actor = actors[0]
    return HeroTraceSummary(
        case_id=str(manifest["case_id"]),
        window_key=str(source_key),
        duration_s=float(s0["duration_s"]),
        minimum_acceleration_mps2=float(s0["minimum_acceleration_mps2"]),
        minimum_jerk_mps3=float(s0["minimum_jerk_mps3"]),
        s1_hmm_posterior=float(s1["scores"]["hmm_posterior"]),
        candidate_actors=int(s2["candidate_actors_within_60m"]),
        top_actor=str(top_actor["alias"]),
        top_actor_score=float(top_actor["score"]),
        model_label=str(parsed["scenario_classification"]),
        model_confidence=float(parsed["confidence_score"]),
        human_label=str(human["scenario_label"]),
        human_primary_actor=str(human["primary_trigger"]),
        retrieval_rank=int(s7["rank"]),
        retrieval_hybrid_score=float(s7["hybrid_score"]),
        checksums_verified=verified,
    )


def format_hero_trace(summary: HeroTraceSummary) -> str:
    """Format a validated trace for a terminal or CI log."""
    return "\n".join(
        [
            "Signals to Semantics - verified real hero trace",
            "=" * 50,
            f"PASS  Integrity    {summary.checksums_verified} published checksums verified",
            f"PASS  Identity     {summary.window_key}",
            (
                "WHEN Event       "
                f"{summary.duration_s:.2f} s; min acceleration "
                f"{summary.minimum_acceleration_mps2:.3f} m/s^2; "
                f"HMM {summary.s1_hmm_posterior:.6f}"
            ),
            (
                "WHO  Actor       "
                f"{summary.top_actor} ranked first among {summary.candidate_actors} candidates "
                f"(score {summary.top_actor_score:.3f})"
            ),
            (
                "WHAT Semantics   "
                f"model={summary.model_label} ({summary.model_confidence:.2f}); "
                f"human={summary.human_label}"
            ),
            f"PASS  Attribution {summary.human_primary_actor} accepted by human review",
            (
                "FIND Retrieval   "
                f"rank {summary.retrieval_rank}; hybrid score "
                f"{summary.retrieval_hybrid_score:.3f}"
            ),
            "=" * 50,
            "PASS  Recorded S0-S7 trace is internally consistent.",
            "      No dataset, model, database, network, or LLM call was used.",
        ]
    )


def run_synthetic_signal_demo() -> dict[str, Any]:
    """Execute the released braking detector on a deterministic synthetic signal."""
    timestamps_s = np.arange(0.0, 8.0 + 0.05, 0.05, dtype=np.float64)
    speed_mps = np.where(
        timestamps_s < 2.0,
        15.0,
        np.where(timestamps_s <= 4.0, 15.0 - 2.0 * (timestamps_s - 2.0), 11.0),
    )
    segments, acceleration, jerk = detect_brakes(
        speed=speed_mps,
        t_s=timestamps_s,
        a_min=-1.0,
        j_min=-2.0,
        min_dur_s=0.25,
        max_gap_s=0.15,
        smooth_window=9,
        smooth_poly=2,
        mode="OR",
        min_delta_v=0.5,
    )
    if not segments:
        raise DemoValidationError("Synthetic braking signal produced no event")

    events = []
    for start, end in segments:
        t_start = float(timestamps_s[start])
        t_end = float(timestamps_s[end])
        events.append(
            {
                "window_key": canonical_window_key("synthetic-braking-demo", t_start, t_end),
                "t_start_s": t_start,
                "t_end_s": t_end,
                "duration_s": t_end - t_start,
                "speed_drop_mps": float(speed_mps[start] - speed_mps[end]),
                "minimum_acceleration_mps2": float(acceleration[start : end + 1].min()),
                "minimum_jerk_mps3": float(jerk[start : end + 1].min()),
            }
        )

    return {
        "demo": "synthetic-signal-detection",
        "sampling_hz": 20.0,
        "sample_count": int(timestamps_s.size),
        "thresholds": {"acceleration_mps2": -1.0, "jerk_mps3": -2.0},
        "events": events,
        "scope": "Executes S0/S1A detection only; no learned outputs are simulated.",
    }


def format_synthetic_demo(result: dict[str, Any]) -> str:
    """Format the synthetic computation result for terminal output."""
    lines = [
        "Signals to Semantics - executable synthetic signal demo",
        "=" * 55,
        f"PASS  Generated {result['sample_count']} samples at {result['sampling_hz']:.0f} Hz",
    ]
    for index, event in enumerate(result["events"], start=1):
        lines.append(
            f"EVENT {index}  {event['t_start_s']:.2f}-{event['t_end_s']:.2f} s; "
            f"duration {event['duration_s']:.2f} s; speed drop {event['speed_drop_mps']:.2f} m/s"
        )
        lines.append(
            f"         min acceleration {event['minimum_acceleration_mps2']:.3f} m/s^2; "
            f"min jerk {event['minimum_jerk_mps3']:.3f} m/s^3"
        )
        lines.append(f"         key {event['window_key']}")
    lines.extend(["=" * 55, f"PASS  {result['scope']}"])
    return "\n".join(lines)
