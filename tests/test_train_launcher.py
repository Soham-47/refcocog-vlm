import unittest
from pathlib import Path


class TrainLauncherTests(unittest.TestCase):
    def test_modelscope_and_huggingface_caches_are_scratch_paths_before_swift(self):
        launcher = (Path(__file__).parents[1] / "slurm" / "train_ablation.sh").read_text()

        modelscope_cache = 'export MODELSCOPE_CACHE="$ROOT/cache/modelscope"'
        huggingface_cache = 'export HF_HOME="$ROOT/cache/huggingface"'
        first_swift_command = launcher.index("swift.cli")

        self.assertIn(modelscope_cache, launcher)
        self.assertIn(huggingface_cache, launcher)
        self.assertLess(launcher.index(modelscope_cache), first_swift_command)
        self.assertLess(launcher.index(huggingface_cache), first_swift_command)

    def test_uses_chat_format_dataset_for_grpo_and_sft_dataset_for_sft(self):
        launcher = (Path(__file__).parents[1] / "slurm" / "train_ablation.sh").read_text()

        self.assertIn('TRAIN="$ROOT/data/refcocog_train500_grpo_v2.jsonl"', launcher)
        self.assertIn('--dataset "$ROOT/data/refcocog_train500_sft_v2.jsonl"', launcher)


if __name__ == "__main__":
    unittest.main()
