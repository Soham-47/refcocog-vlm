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
3. Run baseline evaluation through Slurm.
4. Add and evaluate the RL/GRPO training loop.
