#!/usr/bin/env python3
"""Generate recruiter-facing figures from the verified thesis-result ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
MANIFEST = ASSETS / "figures_manifest.json"
SOURCE_PATHS = [
    ROOT / "results" / "verified" / "final_defense_metrics.json",
    ROOT / "results" / "verified" / "s5_prompt_ablation_val50.csv",
    ROOT / "results" / "verified" / "s7_strict_gt_top10_val50.csv",
]
OUTPUT_NAMES = [
    "architecture_overview.png",
    "architecture_overview.svg",
    "evaluation_funnel.png",
    "evaluation_funnel.svg",
    "results_dashboard.png",
    "results_dashboard.svg",
    "social_preview.png",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _verified_data() -> dict[str, Any]:
    metrics = json.loads(SOURCE_PATHS[0].read_text(encoding="utf-8"))
    return {
        "metrics": metrics,
        "s5": _load_csv(SOURCE_PATHS[1]),
        "s7": _load_csv(SOURCE_PATHS[2]),
    }


def _configure_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            'Figure generation requires the visualization extras: pip install -e ".[viz]"'
        ) from exc

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )
    return plt


def _save(plt: Any, figure: Any, stem: str, *, png_only: bool = False) -> None:
    figure.savefig(
        ASSETS / f"{stem}.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
        metadata={"Software": "Signals to Semantics figure generator"},
    )
    if not png_only:
        svg_path = ASSETS / f"{stem}.svg"
        figure.savefig(
            svg_path,
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
            metadata={"Date": None, "Creator": "Signals to Semantics figure generator"},
        )
        # Matplotlib emits spaces before line endings in SVG path data. Remove
        # them so Git's whitespace checks stay useful and generated diffs clean.
        svg_text = svg_path.read_text(encoding="utf-8")
        svg_path.write_text(
            "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
            encoding="utf-8",
        )
    plt.close(figure)


def _architecture_figure(plt: Any) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    navy = "#102A43"
    colors = ["#D9EAF7", "#DDF3EC", "#FFF0D5", "#E8E2F5"]
    phases = [
        ("PHASE I", "S0-S1", "Event localization", "Signals -> verified windows"),
        ("PHASE II", "S2-S3", "Actor attribution", "Representations -> Top-3"),
        ("PHASE III", "S4-S5", "Semantic reasoning", "Evidence -> label + rationale"),
        ("PHASE IV", "S6-S7", "Scenario retrieval", "Knowledge base -> ranked cases"),
    ]
    fig, ax = plt.subplots(figsize=(15.5, 4.2))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.text(
        0.2,
        3.72,
        "From Signals to Semantics",
        fontsize=21,
        fontweight="bold",
        color=navy,
        va="center",
    )
    ax.text(
        0.2,
        3.27,
        "A traceable validation pipeline from vehicle response to searchable scenario knowledge",
        fontsize=11.5,
        color="#52616B",
        va="center",
    )

    box_width = 3.18
    box_height = 2.05
    starts = [0.2, 4.05, 7.9, 11.75]
    for index, ((phase, stages, title, output), x) in enumerate(zip(phases, starts)):
        box = FancyBboxPatch(
            (x, 0.55),
            box_width,
            box_height,
            boxstyle="round,pad=0.035,rounding_size=0.12",
            linewidth=1.5,
            edgecolor=navy,
            facecolor=colors[index],
        )
        ax.add_patch(box)
        ax.text(x + 0.22, 2.28, phase, fontsize=9, fontweight="bold", color="#486581")
        ax.text(x + 0.22, 1.91, stages, fontsize=16, fontweight="bold", color=navy)
        ax.text(x + 0.22, 1.48, title, fontsize=12.5, fontweight="bold", color=navy)
        ax.text(x + 0.22, 0.93, output, fontsize=10.5, color="#334E68")
        if index < len(starts) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + box_width + 0.10, 1.58),
                    (starts[index + 1] - 0.10, 1.58),
                    arrowstyle="-|>",
                    mutation_scale=18,
                    linewidth=1.8,
                    color="#3E7CB1",
                )
            )
    _save(plt, fig, "architecture_overview")


def _funnel_figure(plt: Any, metrics: dict[str, Any]) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    held_out = metrics["held_out_evaluation"]
    s1 = held_out["s1"]
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Held-out Val50 evaluation funnel", loc="left", color="#102A43", pad=16)
    ax.text(
        0.1,
        4.42,
        "Strict evaluation remains separate from physically verified candidate discovery.",
        color="#52616B",
    )

    boxes = [
        (0.15, 2.05, 2.45, 1.55, "50", "held-out logs", "#D9EAF7"),
        (3.35, 2.05, 2.45, 1.55, str(s1["retained_windows"]), "retained windows", "#DDF3EC"),
        (7.0, 2.95, 2.55, 1.25, str(s1["gt_aligned_windows"]), "GT-aligned\nstrict scoring", "#D9EAF7"),
        (
            7.0,
            0.82,
            2.55,
            1.25,
            str(s1["unlabelled_candidate_discoveries"]),
            "unlabelled\nreview candidates",
            "#FFF0D5",
        ),
    ]
    for x, y, width, height, value, label, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.10",
            facecolor=color,
            edgecolor="#102A43",
            linewidth=1.4,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height * 0.63, value, ha="center", va="center", fontsize=21, fontweight="bold", color="#102A43")
        ax.text(x + width / 2, y + height * 0.26, label, ha="center", va="center", fontsize=10.5, color="#334E68")

    for start, end in [((2.65, 2.83), (3.25, 2.83)), ((5.85, 2.83), (6.9, 3.5)), ((5.85, 2.65), (6.9, 1.45))]:
        ax.add_patch(
            FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color="#3E7CB1")
        )

    ax.text(10.15, 3.52, "Quantitative evidence", fontsize=11.5, fontweight="bold", color="#102A43")
    ax.text(10.15, 3.08, "Used for reported metrics", fontsize=10.5, color="#52616B")
    ax.text(10.15, 1.46, "Discovery queue", fontsize=11.5, fontweight="bold", color="#102A43")
    ax.text(10.15, 0.98, "Not automatically labelled novel", fontsize=10.5, color="#52616B")
    _save(plt, fig, "evaluation_funnel")


def _results_dashboard(plt: Any, data: dict[str, Any]) -> None:
    metrics = data["metrics"]["held_out_evaluation"]
    s5_rows = data["s5"]
    s7_rows = data["s7"]
    blue = "#2878B5"
    teal = "#3A9D8F"
    orange = "#F2A541"
    navy = "#102A43"

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), gridspec_kw={"wspace": 0.38})
    fig.suptitle("Final held-out thesis results", x=0.06, ha="left", fontsize=19, fontweight="bold", color=navy)
    fig.text(0.06, 0.91, "Auditable values from the final Val50 metric ledger", color="#52616B", fontsize=11)

    ax = axes[0]
    rank_counts = [metrics["s3"]["rank_1"]["count"], 1, 2, 0]
    labels = ["Rank 1", "Rank 2", "Rank 3", "Beyond 3"]
    bars = ax.bar(labels, rank_counts, color=[blue, teal, orange, "#BCCCDC"])
    ax.set_title("S3: responsible actor rank", loc="left")
    ax.set_ylabel("GT-aligned windows")
    ax.set_ylim(0, 26)
    ax.grid(axis="y", alpha=0.18)
    for bar, count in zip(bars, rank_counts):
        rate = count / 26 * 100
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.6, f"{count}\n({rate:.1f}%)", ha="center", va="bottom", fontsize=9.5)
    ax.text(0.02, -0.23, "Top-3 containment: 26/26 (100%)", transform=ax.transAxes, color="#334E68", fontsize=10)

    ax = axes[1]
    prompt_order = ["ALLINONE", "CoT", "CP", "Base Prompt"]
    rows_by_prompt = {row["prompt_strategy"]: row for row in s5_rows}
    accuracy = [float(rows_by_prompt[name]["accuracy"]) for name in prompt_order]
    macro_f1 = [float(rows_by_prompt[name]["macro_f1"]) for name in prompt_order]
    y = list(range(len(prompt_order)))
    ax.barh([value + 0.18 for value in y], accuracy, height=0.34, color=blue, label="Accuracy")
    ax.barh([value - 0.18 for value in y], macro_f1, height=0.34, color=teal, label="Macro-F1")
    ax.set_yticks(y, prompt_order)
    ax.set_xlim(0, 1)
    ax.set_title("S5: prompt ablation", loc="left")
    ax.set_xlabel("Score")
    ax.grid(axis="x", alpha=0.18)
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.02, -0.23, "Selected: Base Prompt + gpt-5-chat", transform=ax.transAxes, color="#334E68", fontsize=10)

    ax = axes[2]
    scenario_names = [row["scenario_type"].replace("_", " ") for row in s7_rows]
    p10 = [float(row["p_at_10"]) for row in s7_rows]
    r10 = [float(row["r_at_10"]) for row in s7_rows]
    y = list(range(len(scenario_names)))
    ax.barh([value + 0.18 for value in y], p10, height=0.34, color=blue, label="P@10")
    ax.barh([value - 0.18 for value in y], r10, height=0.34, color=orange, label="R@10")
    ax.set_yticks(y, scenario_names)
    ax.set_xlim(0, 1)
    ax.set_title("S7: strict GT-only retrieval", loc="left")
    ax.set_xlabel("Score")
    ax.grid(axis="x", alpha=0.18)
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.02, -0.23, "K=10; unlabelled windows are not positives", transform=ax.transAxes, color="#334E68", fontsize=10)

    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.06, right=0.98)
    _save(plt, fig, "results_dashboard")


def _social_preview(plt: Any, metrics: dict[str, Any]) -> None:
    held_out = metrics["held_out_evaluation"]
    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.set_facecolor("#102A43")
    fig.patch.set_facecolor("#102A43")
    ax.text(0.75, 5.42, "SIGNALS TO SEMANTICS", color="white", fontsize=30, fontweight="bold")
    ax.text(0.78, 4.83, "Traceable autonomous-driving scenario mining", color="#B9D9EB", fontsize=17)
    ax.text(0.78, 4.38, "PU learning  |  GATv2  |  constrained LLM  |  semantic retrieval", color="#D9EAF7", fontsize=11.5)

    cards = [
        ("56/56", "events retained"),
        ("26/26", "actors in Top-3"),
        (f"{held_out['s5']['macro_f1']:.3f}", "S5 Macro-F1"),
        ("S0-S7", "traceable pipeline"),
    ]
    for index, (value, label) in enumerate(cards):
        x = 0.78 + index * 3.0
        ax.text(x, 3.05, value, color="#4FD1C5", fontsize=24, fontweight="bold")
        ax.text(x, 2.63, label, color="white", fontsize=10.5)
    ax.plot([0.78, 12.0], [2.10, 2.10], color="#486581", linewidth=1.2)
    ax.text(0.78, 1.47, "Hariharan Chandrasekaran", color="white", fontsize=16, fontweight="bold")
    ax.text(0.78, 1.02, "Master's thesis research artifact  |  Argoverse 2  |  offline validation", color="#B9D9EB", fontsize=11.5)
    fig.savefig(
        ASSETS / "social_preview.png",
        dpi=100,
        facecolor=fig.get_facecolor(),
        metadata={"Software": "Signals to Semantics figure generator"},
    )
    plt.close(fig)


def _write_manifest() -> None:
    manifest = {
        "schema_version": 1,
        "generation_script": _relative(Path(__file__)),
        "metric_policy": {
            "s3_hit_at_1": "23/26 = 0.884615; count-derived value approved for public reporting",
            "s5_macro_f1": "0.621; final selected Base Prompt + gpt-5-chat value",
        },
        "thesis_lineage": {
            "architecture_overview": "Redesigned from the system organization in thesis Figure 4.1.",
            "evaluation_funnel": "Redesigned from thesis Figure 6.1.",
            "results_dashboard": "Regenerated from the data underlying thesis Figure 6.6, Figure 6.11, and Table 6.11.",
            "social_preview": "Repository social card assembled from the verified final metric ledger.",
        },
        "sources": [
            {"path": _relative(path), "sha256": _sha256(path)} for path in SOURCE_PATHS
        ],
        "outputs": [
            {"path": f"assets/{name}", "sha256": _sha256(ASSETS / name)}
            for name in OUTPUT_NAMES
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def generate() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    data = _verified_data()
    plt = _configure_matplotlib()
    _architecture_figure(plt)
    _funnel_figure(plt, data["metrics"])
    _results_dashboard(plt, data)
    _social_preview(plt, data["metrics"])
    _write_manifest()
    print(f"Generated {len(OUTPUT_NAMES)} figure files and {MANIFEST.relative_to(ROOT)}")


def check() -> list[str]:
    errors: list[str] = []
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Cannot read {MANIFEST.relative_to(ROOT)}: {exc}"]
    for kind in ("sources", "outputs"):
        for item in manifest.get(kind, []):
            path = ROOT / item["path"]
            if not path.is_file():
                errors.append(f"Missing figure {kind[:-1]}: {item['path']}")
            elif _sha256(path) != item["sha256"]:
                errors.append(f"Changed figure {kind[:-1]}: {item['path']}")
    expected_outputs = {f"assets/{name}" for name in OUTPUT_NAMES}
    recorded_outputs = {item["path"] for item in manifest.get("outputs", [])}
    if recorded_outputs != expected_outputs:
        errors.append("Figure manifest output set is incomplete")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify sources and generated files without Matplotlib.")
    args = parser.parse_args()
    if args.check:
        errors = check()
        if errors:
            print("Figure verification failed:")
            for error in errors:
                print(f"  - {error}")
            return 1
        print("Figure sources and generated outputs match the manifest.")
        return 0
    try:
        generate()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
