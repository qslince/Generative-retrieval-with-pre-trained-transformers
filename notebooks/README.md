# Notebooks

Run notebooks from the repository root so relative paths like `data/`, `src/`,
and `experiments/` resolve consistently. If a Jupyter kernel starts inside this
directory, run `%cd ..` before executing path-sensitive cells.

## Collections

- `original/`: the four notebooks from the upstream repository.
- `coursework/`: the expanded coursework notebook tree from
  `plum_sid_coursework`.

## Original Workflow

1. `original/data_prep_diff_emb.ipynb`
2. `original/RQVAE_train_stage.ipynb`
3. `original/RQVAE_AntiContrastive_train.ipynb`
4. `original/GPT2_Rec_Analysis.ipynb`

## Coursework Workflow

1. `coursework/00_baseline/`: data preparation and original baselines.
2. `coursework/01_sid_generation/`: RQ-VAE training, SID generation, and
   collision handling.
3. `coursework/02_recsys/`: GPT2Rec training/evaluation notebooks.
4. `coursework/03_analysis/`: final experiment analysis, tie-break comparisons,
   and report tables.

Some coursework notebooks import `sid_utils` from `src/`. Run from the
repository root or add `src/` to `PYTHONPATH`.
