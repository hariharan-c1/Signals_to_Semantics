import csv
import hashlib
import json
import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _read_ids(name):
    path = ROOT / "configs" / "splits" / name
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


class RepositoryContractTests(unittest.TestCase):
    def test_split_counts_and_relationships(self):
        dev100 = _read_ids("dev100.txt")
        train550 = _read_ids("train550.txt")
        train650 = _read_ids("train650.txt")
        val50 = _read_ids("val50.txt")

        self.assertEqual(len(dev100), 100)
        self.assertEqual(len(set(dev100)), 100)
        self.assertEqual(len(train550), 550)
        self.assertEqual(len(set(train550)), 550)
        self.assertEqual(len(train650), 650)
        self.assertEqual(len(set(train650)), 650)
        self.assertEqual(len(val50), 50)
        self.assertEqual(len(set(val50)), 50)
        self.assertTrue(set(dev100).isdisjoint(train550))
        self.assertEqual(set(train650), set(dev100) | set(train550))
        self.assertTrue(set(train650).isdisjoint(val50))

    def test_public_configs_parse(self):
        for path in sorted((ROOT / "configs").rglob("*.yaml")):
            with self.subTest(path=path):
                parsed = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
                self.assertIsNotNone(parsed)

    def test_json_examples_and_summaries_parse(self):
        paths = [
            ROOT / "examples" / "evidence" / "synthetic_window.json",
            ROOT / "examples" / "llm_output" / "synthetic_window.llm.json",
            ROOT / "examples" / "hero_scenario" / "manifest.json",
            ROOT / "examples" / "hero_scenario" / "s0_candidate_window.json",
            ROOT / "examples" / "hero_scenario" / "s1_verification.json",
            ROOT / "examples" / "hero_scenario" / "s2_representation_summary.json",
            ROOT / "examples" / "hero_scenario" / "s3_actor_ranking.json",
            ROOT / "examples" / "hero_scenario" / "s4_evidence_pack.json",
            ROOT / "examples" / "hero_scenario" / "s5_llm_output.json",
            ROOT / "examples" / "hero_scenario" / "human_validation.json",
            ROOT / "examples" / "hero_scenario" / "s7_retrieval_trace.json",
            ROOT / "results" / "verified" / "final_defense_metrics.json",
            ROOT / "results" / "verified" / "s1_s3_val50.json",
            ROOT / "results" / "verified" / "s6_integrity_val50.json",
            ROOT / "schemas" / "evidence.schema.json",
            ROOT / "schemas" / "llm_output.schema.json",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assertIsInstance(json.loads(path.read_text()), dict)

    def test_prompt_templates_have_scenario_placeholder(self):
        prompt_dir = ROOT / "scripts" / "s5" / "prompts"
        prompts = sorted(prompt_dir.glob("*.txt"))
        self.assertEqual(len(prompts), 5)
        for path in prompts:
            with self.subTest(path=path):
                text = path.read_text()
                self.assertIn("<<<SCENARIO_YAML>>>", text)
                self.assertIsNone(UUID.search(text))

    def test_final_defense_metric_contract(self):
        path = ROOT / "results" / "verified" / "final_defense_metrics.json"
        metrics = json.loads(path.read_text())
        held_out = metrics["held_out_evaluation"]

        self.assertEqual(held_out["split"], "val50")
        self.assertEqual(held_out["logs"], 50)
        self.assertEqual(held_out["s1"]["retained_windows"], 56)
        self.assertEqual(held_out["s1"]["gt_aligned_windows"], 26)
        self.assertEqual(
            held_out["s1"]["unlabelled_candidate_discoveries"],
            30,
        )
        self.assertEqual(held_out["s3"]["rank_1"]["count"], 23)
        self.assertEqual(held_out["s3"]["within_rank_3"]["count"], 26)
        self.assertEqual(held_out["s5"]["evaluation_logs"], 25)
        self.assertAlmostEqual(held_out["s5"]["accuracy"], 0.760)
        self.assertAlmostEqual(held_out["s5"]["macro_f1"], 0.621)

    def test_final_s5_prompt_ablation(self):
        scoreboard = ROOT / "results" / "verified" / "s5_prompt_ablation_val50.csv"
        with scoreboard.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 4)
        selected = [row for row in rows if row["selected"] == "true"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["prompt_strategy"], "Base Prompt")
        self.assertEqual(selected[0]["model_label"], "gpt-5-chat")
        self.assertEqual(int(selected[0]["n_gt_logs"]), 25)
        self.assertAlmostEqual(float(selected[0]["accuracy"]), 0.760)
        self.assertAlmostEqual(float(selected[0]["macro_f1"]), 0.621)

    def test_final_s7_strict_gt_top10(self):
        metrics = ROOT / "results" / "verified" / "s7_strict_gt_top10_val50.csv"
        with metrics.open(newline="") as handle:
            rows = {row["scenario_type"]: row for row in csv.DictReader(handle)}

        self.assertEqual(
            set(rows),
            {"cut_in", "approach_stop", "obj_crossing", "lead_brake", "ped_crossing"},
        )
        self.assertAlmostEqual(float(rows["obj_crossing"]["p_at_10"]), 0.42)
        self.assertAlmostEqual(float(rows["obj_crossing"]["r_at_10"]), 0.63)
        self.assertAlmostEqual(float(rows["ped_crossing"]["r_at_10"]), 1.00)

    def test_hero_scenario_trace(self):
        hero = ROOT / "examples" / "hero_scenario"
        s0 = json.loads((hero / "s0_candidate_window.json").read_text())
        s1 = json.loads((hero / "s1_verification.json").read_text())
        s2 = json.loads((hero / "s2_representation_summary.json").read_text())
        s3 = json.loads((hero / "s3_actor_ranking.json").read_text())
        s4 = json.loads((hero / "s4_evidence_pack.json").read_text())
        s5 = json.loads((hero / "s5_llm_output.json").read_text())
        human = json.loads((hero / "human_validation.json").read_text())
        s7 = json.loads((hero / "s7_retrieval_trace.json").read_text())

        key = s1["window_key"]
        self.assertEqual(s0["log_id"], key.split("|", 1)[0])
        self.assertEqual(s3["window_key"], key)
        self.assertEqual(s4["window_key"], key)
        self.assertEqual(s5["parsed_result"]["ego_window_key"], key)
        self.assertEqual(human["window_key"], key)
        self.assertEqual(s7["canonical_source_window_key"], key)
        self.assertEqual(s7["database_timestamp_precision_decimals"], 6)
        expected_database_key = "|".join(
            [
                s0["log_id"],
                f'{s0["t_start"]:.6f}',
                f'{s0["t_end"]:.6f}',
            ]
        )
        self.assertEqual(s7["matched_database_window_key"], expected_database_key)
        self.assertEqual(s0["duration_s"], 2.25)
        self.assertAlmostEqual(s1["scores"]["hmm_posterior"], 0.9986784334)
        self.assertEqual(s2["candidate_actors_within_60m"], 9)
        self.assertEqual(s2["learned_embedding_dimensions"], 128)
        self.assertEqual(s3["actors"][0]["alias"], "ACTOR1")
        self.assertAlmostEqual(s3["actors"][0]["score"], 1.0447728634)
        self.assertEqual(s5["parsed_result"]["scenario_classification"], "obj_crossing")
        self.assertEqual(s5["parsed_result"]["primary_responsible_actor"], "ACTOR1")
        self.assertAlmostEqual(s5["parsed_result"]["confidence_score"], 0.83)
        self.assertEqual(human["scenario_label"], "cut_in")
        self.assertEqual(human["primary_trigger"], "ACTOR1")
        self.assertEqual(s7["rank"], 3)
        self.assertAlmostEqual(s7["hybrid_score"], 0.7242108767)

    def test_hero_manifest_checksums(self):
        hero = ROOT / "examples" / "hero_scenario"
        manifest = json.loads((hero / "manifest.json").read_text())
        for artifact in manifest["published_files"]:
            path = hero / artifact["path"]
            with self.subTest(path=path):
                self.assertTrue(path.is_file())
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, artifact["sha256"])

    def test_generated_figure_manifest(self):
        manifest_path = ROOT / "assets" / "figures_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        expected_outputs = {
            "assets/architecture_overview.png",
            "assets/architecture_overview.svg",
            "assets/evaluation_funnel.png",
            "assets/evaluation_funnel.svg",
            "assets/results_dashboard.png",
            "assets/results_dashboard.svg",
            "assets/social_preview.png",
        }
        self.assertEqual(
            {item["path"] for item in manifest["outputs"]},
            expected_outputs,
        )
        self.assertIn("23/26", manifest["metric_policy"]["s3_hit_at_1"])
        self.assertIn("0.621", manifest["metric_policy"]["s5_macro_f1"])
        for kind in ("sources", "outputs"):
            for artifact in manifest[kind]:
                path = ROOT / artifact["path"]
                with self.subTest(kind=kind, path=path):
                    self.assertTrue(path.is_file())
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    self.assertEqual(digest, artifact["sha256"])

    def test_recruiter_contact_and_demo_entry_points(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn(
            "https://www.linkedin.com/in/hariharan-chandrasekaran-/",
            readme,
        )
        self.assertIn("mailto:hariharan.chandrasekaran25@gmail.com", readme)
        self.assertIn("sts demo", readme)
        self.assertIn("sts demo-synthetic", readme)

    def test_relative_markdown_links_exist(self):
        for path in sorted(ROOT.rglob("*.md")):
            for target in MARKDOWN_LINK.findall(path.read_text()):
                if target.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                clean_target = target.split("#", 1)[0]
                with self.subTest(source=path, target=target):
                    self.assertTrue((path.parent / clean_target).exists())


if __name__ == "__main__":
    unittest.main()
