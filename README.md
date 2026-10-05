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
