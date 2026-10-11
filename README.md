# RefCOCOg VLM grounding

This repository contains the code for evaluating and training Qwen3-VL on
2D referring-expression grounding. It prepares RefCOCOg image manifests,
converts them to ms-swift data, runs SFT or GRPO jobs on HPC, and evaluates
predicted boxes with IoU metrics.

The code is experiment code rather than a packaged training framework. Images,
manifests, model caches, checkpoints, and prediction files stay outside Git.
The checked-in Slurm launcher assumes the shared layout under
`/scratch/ambpdc/soham/vlm`.

## Experiment summary

The completed paired diagnostic used Qwen3-VL-8B-Instruct with four systems:

- `base`: the base model.
- `sft`: a LoRA supervised fine-tuning run.
- `grpo`: direct GRPO from the base model.
- `sft_grpo`: GRPO initialized from the SFT adapter.

The runs used the corrected `refcocog_train500_v2.jsonl` and
`refcocog_validation200_v2.jsonl` manifests. They contain 500 and 200 rows,
respectively, with zero overlapping reference IDs. All paired comparisons in
the report are comparable. The validation set is a small diagnostic subset,
not an official benchmark or leaderboard evaluation.

### Validation, 200 examples

| System | Mean IoU | Acc@0.5 | Acc@0.75 | Parse failures |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.8023357 | 0.845 | 0.780 | 1 |
| SFT | 0.8160114 | 0.860 | 0.800 | 0 |
| Direct GRPO | 0.8042084 | 0.850 | 0.780 | 1 |
| SFT then GRPO | 0.8158485 | 0.860 | 0.795 | 0 |

Paired mean-IoU changes against the base run were +0.01367575 for SFT,
+0.00187275 for direct GRPO, and +0.01351283 for SFT then GRPO. The
corresponding improved, degraded, and unchanged counts were (101, 97, 2),
(44, 44, 112), and (99, 99, 2).

### Training, 500 examples

| System | Mean IoU | Acc@0.5 | Acc@0.75 | Parse failures |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.7976700 | 0.858 | 0.744 | 0 |
| SFT | 0.8251656 | 0.890 | 0.782 | 0 |
| Direct GRPO | 0.8019457 | 0.864 | 0.748 | 0 |
| SFT then GRPO | 0.8291169 | 0.890 | 0.794 | 0 |

Paired mean-IoU changes against the base run were +0.02749562 for SFT,
+0.00427569 for direct GRPO, and +0.03144688 for SFT then GRPO. The
corresponding improved, degraded, and unchanged counts were (258, 237, 5),
(105, 96, 299), and (278, 216, 6).

The full report is [reports/refcocog_grpo_diagnostic.tex](reports/refcocog_grpo_diagnostic.tex).
It records the setup, training runs, paired analysis, limitations, and the
HPC artifact paths.

## Quick start

Install the lightweight local dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The repository does not install PyTorch, Transformers, PEFT, or ms-swift.
Use the existing model environment for evaluation and Slurm training.

Run the dependency-light checks from the repository root:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 scripts/evaluate_grounding.py --self-check
python3 scripts/analyze_grounding_diagnostic.py --self-check
python3 -m compileall -q src scripts
bash -n slurm/train_ablation.sh
```

## Data preparation

Cache the images and write a manifest with
`scripts/prepare_refcocog_images.py`. The downloader tries the dataset Flickr
URL, the COCO URL, and the current COCO host. It verifies existing images and
skips examples that cannot be downloaded.

```bash
ROOT=/scratch/ambpdc/soham/vlm
python3 scripts/prepare_refcocog_images.py \
  --split 'train[:500]' \
  --output-dir "$ROOT/coco" \
  --manifest "$ROOT/data/refcocog_train500_manifest.jsonl"
```

Convert one manifest into the GRPO and SFT JSONL formats:

```bash
python3 scripts/prepare_swift_data.py \
  --manifest "$ROOT/data/refcocog_train500_manifest.jsonl" \
  --grpo-output "$ROOT/data/refcocog_train500_grpo_v2.jsonl" \
  --sft-output "$ROOT/data/refcocog_train500_sft_v2.jsonl"
```

The prompt asks for one box as `[[x1, y1, x2, y2]]` on a `0..1000`
scale. Ground-truth boxes are kept in cached-image pixel coordinates for IoU
and the GRPO reward. See
[`src/refcocog/grounding.py`](src/refcocog/grounding.py) for the shared parser,
coordinate conversion, IoU, and reward logic.

## Training

Submit the checked-in launcher from the HPC checkout:

```bash
cd /scratch/ambpdc/soham/vlm/repo
sbatch slurm/train_ablation.sh sft
sbatch slurm/train_ablation.sh grpo_base
sbatch slurm/train_ablation.sh grpo_after_sft
```

The launcher uses Qwen3-VL-8B-Instruct, bfloat16, LoRA rank 8, LoRA alpha 32,
`all-linear` target modules, maximum sequence length 2048, and one example per
device batch. SFT uses one epoch, learning rate `1e-4`, and gradient
accumulation 8. GRPO uses learning rate `5e-6`, four generations, `beta=0.04`,
and maximum completion length 64. `grpo_after_sft` requires the SFT adapter.

Recorded training runs:

| Run | Steps | Runtime | Train loss | Final logged value |
| --- | ---: | ---: | ---: | ---: |
| SFT | 63/63 | 130.4228 s | 0.89258396 | token accuracy 0.69059829 |
| Direct GRPO | 500/500 | 2134.9276 s | 0.00159036 | reward 0.68845888 |
| SFT then GRPO | 500/500 | 2531.4886 s | 0.00034432 | reward 0.69928728 |

These training losses and logged values are not comparable across methods.

## Evaluation and paired diagnostics

Evaluate a model or LoRA adapter on a manifest. An unparseable prediction is
written with `parse_ok=false` and receives IoU `0.0`.

```bash
python3 scripts/evaluate_grounding.py \
  --manifest "$ROOT/data/refcocog_validation200_v2.jsonl" \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --predictions "$ROOT/outputs/eval/base_validation_predictions.jsonl" \
  --summary "$ROOT/outputs/eval/base_validation_metrics.json"
```

Add `--adapter PATH` for a LoRA checkpoint. The summary contains mean IoU,
Acc@0.5, Acc@0.75, and parse failures. The diagnostic script checks that rows,
phrases, and ground-truth boxes match before calculating paired deltas:

```bash
python3 scripts/analyze_grounding_diagnostic.py \
  --prediction base/train="$ROOT/outputs/diagnostic/base_train.jsonl" \
  --prediction sft/train="$ROOT/outputs/diagnostic/sft_train.jsonl" \
  --prediction grpo/train="$ROOT/outputs/diagnostic/grpo_train_h100.jsonl" \
  --prediction sft_grpo/train="$ROOT/outputs/diagnostic/sft_grpo_train_h100.jsonl" \
  --prediction base/val="$ROOT/outputs/diagnostic/base_val.jsonl" \
  --prediction sft/val="$ROOT/outputs/diagnostic/sft_val.jsonl" \
  --prediction grpo/val="$ROOT/outputs/diagnostic/grpo_val.jsonl" \
  --prediction sft_grpo/val="$ROOT/outputs/diagnostic/sft_grpo_val.jsonl" \
  --output "$ROOT/outputs/diagnostic/full_report.json"
```

Use the same manifest for every model in a paired comparison. Earlier
comparisons made with mismatched manifests were superseded by the v2 diagnostic
reported here.

## Repository map and artifacts

- `src/refcocog/grounding.py`: box parsing, coordinate conversion, IoU, and reward.
- `scripts/`: data preparation, evaluation, diagnostic, and dependency checks.
- `slurm/`: the training launcher and GRPO dependency pins.
- `notebooks/`: exploratory dataset checks and visualizations.
- `tests/`: dependency-light geometry and launcher contract tests.

The diagnostic artifacts remain on HPC and are not tracked in this repository:

```text
/scratch/ambpdc/soham/vlm/outputs/diagnostic/
├── base_train.jsonl
├── sft_train.jsonl
├── grpo_train_h100.jsonl
├── sft_grpo_train_h100.jsonl
├── base_val.jsonl
├── sft_val.jsonl
├── grpo_val.jsonl
├── sft_grpo_val.jsonl
└── full_report.json
```

The repository intentionally does not include model weights, COCO images,
manifests, checkpoints, or generated prediction files.
