# RefCOCOg VLM Grounding

Research code for evaluating and improving Qwen3-VL on 2D referring-expression grounding.

## Project layout

- `notebooks/`: exploratory checks and visualizations
- `src/refcocog/`: reusable Python code
- `scripts/`: reproducible experiment entry points
- `slurm/`: HPC job scripts
- `configs/`: experiment settings
- `outputs/`: local experiment outputs; ignored by Git

Datasets, model caches, checkpoints, and outputs stay on the HPC filesystem and are not committed.

## Current workflow

1. Validate RefCOCOg annotations and images in a notebook.
2. Move reusable dataset/model/metric code into `src/refcocog/`.
3. Run a zero-shot baseline on the validation manifest:

   ```bash
   python scripts/evaluate_grounding.py --self-check
   python scripts/evaluate_grounding.py \
     --manifest /scratch/ambpdc/soham/vlm/data/refcocog_validation50.jsonl \
     --predictions /scratch/ambpdc/soham/vlm/outputs/baseline_val50_predictions.jsonl \
     --summary /scratch/ambpdc/soham/vlm/outputs/baseline_val50_metrics.json
   ```

   The script reports mean IoU, Acc@0.5, Acc@0.75, and parse failures. It counts an unparseable prediction as IoU 0.
4. Add and evaluate the RL/GRPO training loop.

## Paired diagnostic evaluation

Use the same manifest for every model in a comparison. The analyzer reports
train/validation metrics, box-shape statistics, and paired per-example wins
and losses. Labels use `model/split`; the `base` prefix is the comparison
reference for each split:

```bash
python scripts/analyze_grounding_diagnostic.py --self-check
python scripts/analyze_grounding_diagnostic.py \
  --prediction base/train=/scratch/ambpdc/soham/vlm/outputs/diagnostic/base_train.jsonl \
  --prediction sft/train=/scratch/ambpdc/soham/vlm/outputs/diagnostic/sft_train.jsonl \
  --prediction grpo/train=/scratch/ambpdc/soham/vlm/outputs/diagnostic/grpo_train.jsonl \
  --prediction sft_grpo/train=/scratch/ambpdc/soham/vlm/outputs/diagnostic/sft_grpo_train.jsonl \
  --prediction base/val=/scratch/ambpdc/soham/vlm/outputs/diagnostic/base_val.jsonl \
  --prediction sft/val=/scratch/ambpdc/soham/vlm/outputs/diagnostic/sft_val.jsonl \
  --prediction grpo/val=/scratch/ambpdc/soham/vlm/outputs/diagnostic/grpo_val.jsonl \
  --prediction sft_grpo/val=/scratch/ambpdc/soham/vlm/outputs/diagnostic/sft_grpo_val.jsonl \
  --output /scratch/ambpdc/soham/vlm/outputs/diagnostic/report.json
```

The analyzer marks a comparison non-comparable when row keys, phrases, or
ground-truth boxes differ. Use the corrected `refcocog_train500_v2.jsonl` and
`refcocog_validation200_v2.jsonl` manifests for the current experiment.

## GRPO environment preflight

The GRPO launcher checks the pinned GRPO-only dependencies before starting either
GRPO experiment. Install them once in the shared `ms_swift` environment; the
launcher only checks them on later jobs and does not reinstall packages:

```bash
/scratch/ambpdc/conda/envs/ms_swift/bin/python -m pip install \
  -r /scratch/ambpdc/soham/vlm/repo/slurm/grpo_requirements.txt
```
