import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux/Bash launcher tests")
class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="relmllm launch ")
        self.addCleanup(self.tmp.cleanup)
        self.cwd = Path(self.tmp.name)
        self.root = Path(__file__).resolve().parents[1]
        self.calls = self.cwd / "calls"
        self.env = dict(os.environ, CALL_LOG=str(self.calls))
        stub = self.cwd / "stub.sh"
        stub.write_text(
            'python() { printf "python\\0" >> "$CALL_LOG"; '
            'printf "%s\\0" "$@" >> "$CALL_LOG"; }\n'
            'torchrun() { printf "torchrun\\0" >> "$CALL_LOG"; '
            'printf "%s\\0" "$@" >> "$CALL_LOG"; }\n'
        )
        self.env["BASH_ENV"] = str(stub)
        self.data = self.cwd / "input.json"
        self.data.write_text("[]")
        self.model = self.cwd / "model"
        self.model.mkdir()

    def run_script(self, script, args, env=None):
        return subprocess.run(
            ["bash", str(self.root / "scripts" / script), *map(str, args)],
            cwd=self.cwd, env=env or self.env, capture_output=True, text=True,
        )

    def test_eval_routing_and_guards(self):
        for benchmark in ("groundbench", "degraded", "rgbth", "dv1k"):
            with self.subTest(benchmark=benchmark):
                output = self.cwd / benchmark
                data = self.model if benchmark == "rgbth" else self.data
                result = self.run_script("evaluate.sh", [benchmark, self.model, data, output])
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = self.calls.read_bytes().split(b"\0")
                evaluator = "grounding" if benchmark == "degraded" else benchmark
                if evaluator == "groundbench":
                    evaluator = "grounding"
                self.assertIn(f"tools/evaluate_{evaluator}.py".encode(), calls)
                self.assertIn(str(self.model).encode(), calls)
                result = self.run_script("evaluate.sh", [benchmark, self.model, data, output])
                self.assertEqual(result.returncode, 2)
        self.assertEqual(self.run_script("evaluate.sh", []).returncode, 2)
        self.assertEqual(self.run_script("evaluate.sh", ["unknown", self.model, self.data, self.cwd / "bad"]).returncode, 2)
        self.assertEqual(self.run_script("evaluate.sh", ["dv1k", self.model, "absent", self.cwd / "bad"]).returncode, 2)

    def test_training_pipeline_and_dry_run(self):
        args = [self.model, self.data, self.data, self.data, self.data]
        output = self.cwd / "full"
        result = self.run_script("train.sh", [*args, output])
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls.read_bytes().split(b"\0")
        self.assertEqual(calls.count(b"python"), 2)
        self.assertEqual(calls.count(b"torchrun"), 2)
        self.assertLess(calls.index(b"train_sft.py"), calls.index(b"--adapter_path"))
        self.assertIn(b"sequence", calls)
        self.assertIn(str(output / "sft/final_model").encode(), calls)
        self.assertIn(str(output / "gspo/final_model").encode(), calls)
        self.assertIn(b"tools/merge_adapter.py", calls)
        self.assertEqual(self.run_script("train.sh", [*args, output]).returncode, 2)
        self.calls.unlink()
        result = self.run_script("train.sh", [*args, self.cwd / "dry"], dict(self.env, DRY_RUN="1"))
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls.read_bytes().split(b"\0")
        self.assertIn(b"--dry_run_rewards", calls)
        self.assertNotIn(b"torchrun", calls)
        self.assertEqual(self.run_script("train.sh", []).returncode, 2)
