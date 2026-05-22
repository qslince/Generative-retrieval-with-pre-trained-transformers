# Notebooks

Run notebooks from the repository root so relative paths like `data/`, `src/`,
and `experiments/` resolve consistently. If a Jupyter kernel starts inside this
directory, run `%cd ..` before executing path-sensitive cells.

## Coursework Workflow

1. `coursework/00_baseline/`: data preparation and original baselines.
2. `coursework/01_sid_generation/`: RQ-VAE training, SID generation, and
   collision handling.
3. `coursework/02_recsys/`: GPT2Rec training/evaluation notebooks.
4. `coursework/03_analysis/`: final experiment analysis, tie-break comparisons,
   and report tables.

The upstream notebooks are folded into the same coursework tree:

- `coursework/00_baseline/data_prep_diff_emb_upstream.ipynb`
- `coursework/01_sid_generation/RQVAE_train_stage_upstream.ipynb`
- `coursework/01_sid_generation/RQVAE_AntiContrastive_train_upstream.ipynb`

Some coursework notebooks import `sid_utils` from `src/`. Run from the
repository root or add `src/` to `PYTHONPATH`.
