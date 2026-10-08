import unittest
from pathlib import Path
import subprocess
import sys
import tempfile


class TrainLauncherTests(unittest.TestCase):
    launcher = Path(__file__).parents[1] / "slurm" / "train_ablation.sh"

    def test_modelscope_and_huggingface_caches_are_scratch_paths_before_swift(self):
        launcher = self.launcher.read_text()

        modelscope_cache = 'export MODELSCOPE_CACHE="$ROOT/cache/modelscope"'
        huggingface_cache = 'export HF_HOME="$ROOT/cache/huggingface"'
        first_swift_command = launcher.index("swift.cli")

        self.assertIn(modelscope_cache, launcher)
        self.assertIn(huggingface_cache, launcher)
        self.assertLess(launcher.index(modelscope_cache), first_swift_command)
        self.assertLess(launcher.index(huggingface_cache), first_swift_command)

    def test_uses_chat_format_dataset_for_grpo_and_sft_dataset_for_sft(self):
        launcher = self.launcher.read_text()

        self.assertIn('TRAIN="$ROOT/data/refcocog_train500_grpo_v2.jsonl"', launcher)
        self.assertIn('--dataset "$ROOT/data/refcocog_train500_sft_v2.jsonl"', launcher)

    def test_grpo_preflight_runs_before_each_grpo_launch(self):
        launcher = self.launcher.read_text()

        self.assertIn('GRPO_REQUIREMENTS="$REPO/slurm/grpo_requirements.txt"', launcher)
        for experiment in ("grpo_base", "grpo_after_sft"):
            branch = launcher.split(f"  {experiment})", 1)[1].split("\n    ;;", 1)[0]
            self.assertLess(branch.index("grpo_preflight"), branch.index("swift.cli.rlhf"))

    def test_preflight_rejects_an_unavailable_version(self):
        script = self.launcher.parents[1] / "scripts" / "check_grpo_dependencies.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            requirements = Path(tmpdir) / "requirements.txt"
            requirements.write_text("msgspec==0.0.0\n")
            result = subprocess.run(
                [sys.executable, str(script), str(requirements)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("msgspec", result.stderr)

    def test_grpo_requirements_pin_the_hpc_dependencies(self):
        requirements = self.launcher.parent / "grpo_requirements.txt"

        self.assertEqual(
            requirements.read_text().splitlines()[-2:],
            ["msgspec==0.20.0", "eval-type-backport==0.4.0"],
        )


if __name__ == "__main__":
    unittest.main()
