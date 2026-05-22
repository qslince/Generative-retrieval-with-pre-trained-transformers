# File Map

This map reflects the merged repository layout.

## Source Code

- `src/sid_utils.py`: reusable SID assignment, collision handling, SID metrics,
  and Spearman helper functions.

## Original Upstream Notebooks

- `notebooks/original/data_prep_diff_emb.ipynb`: data preparation workflow from
  the upstream repository.
- `notebooks/original/RQVAE_train_stage.ipynb`: main RQ-VAE training workflow.
- `notebooks/original/RQVAE_AntiContrastive_train.ipynb`: anti-contrastive
  RQ-VAE variant.
- `notebooks/original/GPT2_Rec_Analysis.ipynb`: GPT2Rec analysis notebook.

## Coursework Notebooks

- `notebooks/coursework/00_baseline/`: data preparation and original PLUM/RQ-VAE
  baselines.
- `notebooks/coursework/01_sid_generation/`: improved RQ-VAE runs, SID
  generation, and collision disambiguation.
- `notebooks/coursework/02_recsys/`: GPT2Rec training and evaluation over SID
  variants.
- `notebooks/coursework/03_analysis/`: tie-break, multi-seed, Spearman, and
  final analysis notebooks.

## Experiment Artifacts

- `experiments/configs/main_experiments.yaml`: fixed experiment configuration
  and SID variant definitions.
- `experiments/tables/`: full coursework CSV export.
- `experiments/figures/`: full coursework figure export.
- `results/tables/`: compact report-ready summary tables.
- `results/figures/`: compact report-ready figures.

Large checkpoints are intentionally not stored in git.
