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
from train_sft import RelationSFTDataset, RelationSFTCollator
import torch
from types import SimpleNamespace
from unittest.mock import patch
import train as rl
from relmllm.evidence_rewards import compute_hgrpo_modality_evidence_reward, final_evidence_consistency_reward
from tools.build_relation_data import derive_label


class ReleaseTests(unittest.TestCase):
    def test_gspo_defaults(self):
        with patch.object(sys, "argv", ["train.py", "--train_json", "train.json",
                                      "--eval_json", "val.json", "--output_dir", "out"]):
            args = rl.parse_args()
        self.assertEqual((args.importance_sampling_level, args.loss_type, args.scale_rewards),
                         ("sequence", "grpo", "group"))
        self.assertEqual((args.num_generations, args.max_completion_length), (4, 1024))
        self.assertEqual((args.num_train_epochs, args.max_steps), (1, -1))
        self.assertEqual(args.learning_rate, 1e-6)

    def test_hierarchical_reward_equation(self):
        target = [10, 20, 100, 200]
        output = {"bbox": target, "rgb_reliability": "high", "tir_reliability": "medium",
                  "relation": "complementary", "dominant_modality": "RGB", "evidence": "target",
                  "steps": [
                      {"type": "rgb_evidence", "modality": "RGB", "bbox": target,
                       "reliability": "high", "sub_answer": "RGB target evidence"},
                      {"type": "tir_evidence", "modality": "TIR", "bbox": target,
                       "reliability": "medium", "sub_answer": "TIR target evidence"},
                      {"type": "relation_check", "modality": "RGB", "relation": "complementary",
                       "sub_answer": "RGB TIR relation"}]}
        parts = compute_hgrpo_modality_evidence_reward(output, target, "complementary", "high", "medium", "RGB", 640, 480)
        self.assertAlmostEqual(parts["total"], .2 * parts["format"] + .5 * parts["hungarian"] + .3 * parts["final_consistency"])
        geo, iou = rl.geometry_reward(output, target, 640, 480)
        task = .15 * rl.format_reward(output) + .5 * geo + .35 * rl.relative_reward(iou, .8)
        expected = .85 * (.75 * task + .25) + .15 * parts["total"]
        actual = rl.bbox_relative_reward([None], [json.dumps(output)], [target], [640], [480], [.8],
                                        ["complementary"], ["high"], ["medium"], ["RGB"])[0]
        self.assertAlmostEqual(actual, expected)
        output["bbox"] = [300, 300, 400, 400]
        self.assertEqual(final_evidence_consistency_reward(output, target, 1, 640, 480), 0)

    def test_relation_mining_rules(self):
        def pred(iou):
            return {"iou": iou, "format_ok": 1}
        self.assertEqual(derive_label(pred(.8), pred(.05), pred(.1))["relation"], "conflict")
        self.assertEqual(derive_label(pred(.1), pred(.1), pred(.8))["dominant_modality"], "Both")
        self.assertEqual(derive_label(pred(.1), pred(.1), pred(.1))["relation"], "uncertain")

    def test_rl_prompt_evidence_relation_decision_order(self):
        text = rl.build_prompt({"query": "person", "image_width": 640, "image_height": 480})[1]["content"]
        schema = text[text.index('{"steps"'):]
        self.assertLess(schema.index('"steps"'), schema.index('"rgb_reliability"'))
        self.assertLess(schema.index('"dominant_modality"'), schema.rindex('"bbox"'))
        self.assertNotIn("BBox first", text)

    def test_training_data_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (16, 16)).save(root / "rgb.png")
            Image.new("RGB", (16, 16)).save(root / "tir.png")
            row = {"images": ["rgb.png", "tir.png"], "messages": [
                {"role": "system", "content": "JSON only"},
                {"role": "user", "content": "Locate person"},
                {"role": "assistant", "content": '{"bbox":[1,2,3,4]}'}]}
            path = root / "train.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = RelationSFTDataset(path)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["images"][0], str(root / "rgb.png"))
            row["messages"].pop()
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaises(ValueError):
                RelationSFTDataset(path)

    def test_assistant_only_mask(self):
        def pad(rows, **kwargs):
            size = max(len(row["input_ids"]) for row in rows)
            return {key: torch.stack([torch.nn.functional.pad(row[key],
                    (0, size - len(row[key]))) for row in rows])
                    for key in ("input_ids", "attention_mask")}
        class Processor:
            tokenizer = SimpleNamespace(padding_side="right", pad_token_id=0, pad=pad)
            def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
                ids = [1, 2] if add_generation_prompt else [1, 2, 0, 4]
                return {"input_ids": torch.tensor([ids]),
                        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
        collator = RelationSFTCollator(Processor())
        collator._messages_with_images = lambda row, include_answer: []
        result = collator([{}])
        self.assertEqual(result["labels"].tolist(), [[-100, -100, 0, 4]])
        collator.processor.tokenizer.padding_side = "left"
        with self.assertRaises(ValueError):
            RelationSFTCollator(collator.processor)
        collator.processor.tokenizer.padding_side = "right"

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
