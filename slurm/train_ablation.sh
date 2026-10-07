#!/usr/bin/env bash
# Submit with: sbatch slurm/train_ablation.sh sft|grpo_base|grpo_after_sft
#SBATCH --job-name=refcocog
#SBATCH --partition=gpu_h200_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ambpdc/soham/vlm/logs/train_%j.out
#SBATCH --error=/scratch/ambpdc/soham/vlm/logs/train_%j.err

set -euo pipefail

ROOT=/scratch/ambpdc/soham/vlm
REPO="$ROOT/repo"
PYTHON=/scratch/ambpdc/conda/envs/ms_swift/bin/python
OUTPUT="$ROOT/outputs/ablation"
TRAIN="$ROOT/data/refcocog_train500_grpo_v2.jsonl"
PLUGIN="$REPO/scripts/refcocog_iou_reward.py"
MODEL=Qwen/Qwen3-VL-8B-Instruct
EXPERIMENT=${1:?Usage: sbatch slurm/train_ablation.sh sft\|grpo_base\|grpo_after_sft}

export HF_HOME="$ROOT/cache/huggingface"
export MODELSCOPE_CACHE="$ROOT/cache/modelscope"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
mkdir -p "$OUTPUT" "$ROOT/logs"

common=(--model "$MODEL" --train_type lora --torch_dtype bfloat16
        --dataset "$TRAIN" --num_train_epochs 1 --per_device_train_batch_size 1
        --gradient_checkpointing true --gradient_accumulation_steps 4
        --learning_rate 5e-6 --lora_rank 8 --lora_alpha 32 --target_modules all-linear
        --max_length 2048 --save_strategy epoch --save_total_limit 1
        --eval_strategy no --split_dataset_ratio 0 --logging_steps 5
        --dataloader_num_workers 2)

case "$EXPERIMENT" in
  sft)
    "$PYTHON" -m swift.cli.sft \
      --model "$MODEL" --train_type lora --torch_dtype bfloat16 \
      --dataset "$ROOT/data/refcocog_train500_sft_v2.jsonl" \
      --num_train_epochs 1 --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 --gradient_checkpointing true \
      --learning_rate 1e-4 --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
      --max_length 2048 --save_strategy epoch --save_total_limit 1 \
      --eval_strategy no --split_dataset_ratio 0 --logging_steps 5 \
      --dataloader_num_workers 2 --output_dir "$OUTPUT/sft"
    ;;
  grpo_base)
    "$PYTHON" -m swift.cli.rlhf "${common[@]}" \
      --rlhf_type grpo --external_plugins "$PLUGIN" --reward_funcs refcocog_iou \
      --reward_weights 1.0 --num_generations 4 --beta 0.04 \
      --max_completion_length 64 --log_completions true \
      --output_dir "$OUTPUT/grpo_base"
    ;;
  grpo_after_sft)
    SFT_ADAPTER=$(find "$OUTPUT/sft" -type f -name adapter_config.json -print -quit)
    if [[ -z "$SFT_ADAPTER" ]]; then
      echo "No SFT adapter found under $OUTPUT/sft" >&2
      exit 2
    fi
    SFT_ADAPTER=$(dirname "$SFT_ADAPTER")
    "$PYTHON" -m swift.cli.rlhf "${common[@]}" \
      --rlhf_type grpo --external_plugins "$PLUGIN" --reward_funcs refcocog_iou \
      --reward_weights 1.0 --num_generations 4 --beta 0.04 \
      --max_completion_length 64 --log_completions true \
      --adapters "$SFT_ADAPTER" --ref_adapters "$SFT_ADAPTER" \
      --output_dir "$OUTPUT/grpo_after_sft"
    ;;
  *)
    echo "Unknown experiment: $EXPERIMENT" >&2
    exit 2
    ;;
esac
