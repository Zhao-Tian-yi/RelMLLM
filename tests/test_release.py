import importlib.util
import sys
import unittest
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from inference import extract_json, normalized_to_pixels, build_messages
from PIL import Image
from evaluate_grounding import iou_xyxy, valid_bbox
from evaluate_dv1k import answer_type, normalize_answer, score_answer
from evaluate_rgbth import write_report


class ReleaseTests(unittest.TestCase):
    def test_fenced_json(self):
        self.assertEqual(extract_json('Explanation ```json\n{"bbox":[1,2,3,4]}\n```'),
                         {"bbox": [1, 2, 3, 4]})

    def test_strings(self):
        self.assertEqual(extract_json('prefix {"text":"brace } inside"}'),
                         {"text": "brace } inside"})
        self.assertIsNone(extract_json("There are none."))

    def test_pixels_and_iou(self):
        self.assertEqual(normalized_to_pixels([0, 0, 1000, 1000], 640, 480),
                         [0, 0, 640, 480])
        self.assertEqual(iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertIsNone(valid_bbox([2, 0, 1, 3]))

    def test_prompt_and_order(self):
        rgb, tir = Image.new("RGB", (640, 480)), Image.new("RGB", (640, 480))
        messages = build_messages("grounding", "paper", "person", rgb, tir)
        content = messages[1]["content"]
        self.assertIs(content[0]["image"], rgb)
        self.assertIs(content[1]["image"], tir)
        self.assertIn("0..1000", content[2]["text"])
        self.assertIn("640x480", content[2]["text"])

    def test_count(self):
        kind = answer_type("How many people?", "two")
        self.assertEqual(kind, "count")
        value = normalize_answer("2", kind)
        self.assertEqual(score_answer(value, value, kind), 1)

    def test_rgbth_nested_macro(self):
        rows = []
        for kind, skill, values in [("Single RGB Image", "A", [1, 1, 1]),
                                    ("Single RGB Image", "B", [0]),
                                    ("RGB-Thermal Pair", "A", [1]),
                                    ("RGB-Thermal Pair", "B", [1])]:
            rows.extend({"data_type": kind, "skill_type": skill, "correct": value,
                         "valid": 1, "data_source": "synthetic"} for value in values)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            write_report(path, rows)
            summary = json.loads(path.with_suffix(".json").read_text())["summary"]
            self.assertEqual(summary["macro_qacc"], .75)


if __name__ == "__main__":
    unittest.main()
