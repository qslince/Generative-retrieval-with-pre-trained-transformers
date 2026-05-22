# Notebooks

Run these notebooks from the repository root so that relative paths like
`data/` resolve consistently. If your Jupyter kernel starts inside this
directory, run `%cd ..` in a setup cell or update the data paths explicitly.

Recommended order:

1. `data_prep_diff_emb.ipynb`
2. `RQVAE_train_stage.ipynb`
3. `RQVAE_AntiContrastive_train.ipynb`
4. `GPT2_Rec_Analysis.ipynb`

The notebooks still contain some original Kaggle/local path fallbacks. Review
the setup cells before a full training run.
