# Contributing

This repository is a notebook-first research project. Keep changes focused and
easy to review.

## Local Setup

1. Create and activate a Python 3.12 environment.
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Start Jupyter with `jupyter lab`.
4. Run notebooks from the repository root so relative `data/` paths resolve
   consistently.

## Working With Notebooks

- Keep exploratory changes in clearly named cells.
- Avoid committing generated checkpoints, plots, or temporary outputs.
- Clear large notebook outputs before committing unless the output is necessary
  for review.
- Document any changed data paths or hyperparameters in the notebook cell where
  they are configured.

## Data and Artifacts

The tracked `data/` files are part of the project input state. Generated
artifacts such as RQ-VAE checkpoints, GPT checkpoints, and result plots should
stay out of git unless they are intentionally added as release assets.

## Checks Before Commit

- Confirm that modified notebooks open in Jupyter.
- Validate that notebook JSON is readable.
- Re-run only the notebooks affected by the change when data, model, or metric
  logic changes.
- For documentation-only changes, a lightweight JSON-read smoke check is enough.
