import csv
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

    def test_verified_scoreboard_optima(self):
        scoreboard = ROOT / "results" / "verified" / "s6_backend_scoreboard_val50.csv"
        with scoreboard.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 12)
        best_accuracy = max(rows, key=lambda row: float(row["accuracy_canonical_gt"]))
        best_macro_f1 = max(rows, key=lambda row: float(row["macro_f1_canonical_gt"]))
        self.assertEqual(best_accuracy["prompt_type"], "base_prompt")
        self.assertEqual(best_accuracy["model_name"], "gpt-5-chat")
        self.assertAlmostEqual(
            float(best_accuracy["accuracy_canonical_gt"]),
            0.576923,
            places=6,
        )
        self.assertEqual(best_macro_f1["prompt_type"], "CoT")
        self.assertEqual(best_macro_f1["model_name"], "ollama_gpt-oss")
        self.assertAlmostEqual(
            float(best_macro_f1["macro_f1_canonical_gt"]),
            0.601270,
            places=6,
        )

    def test_verified_retrieval_k10(self):
        metrics = ROOT / "results" / "verified" / "s7_macro_micro_val50.csv"
        with metrics.open(newline="") as handle:
            rows = {int(row["K"]): row for row in csv.DictReader(handle)}

        self.assertEqual(set(rows), {1, 3, 5, 10})
        self.assertAlmostEqual(
            float(rows[10]["macro_gt_precision_mean_over_queries"]),
            0.266667,
            places=6,
        )
        self.assertAlmostEqual(
            float(rows[10]["macro_gt_coverage_mean_over_queries"]),
            0.465625,
            places=6,
        )

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
