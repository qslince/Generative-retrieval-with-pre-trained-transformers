# Generative Retrieval with Pre-trained Transformers

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Jupyter](https://img.shields.io/badge/interface-Jupyter-orange.svg)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c.svg)
![License](https://img.shields.io/badge/license-not%20specified-lightgrey.svg)

This repository contains a notebook-based research project for generative
retrieval in sequential recommendation. The workflow builds item
representations, learns residual-quantized semantic identifiers with RQ-VAE
variants, and uses a GPT-style sequence model for recommendation analysis.

The project is intentionally kept close to the original exploratory notebooks.
This documentation pass adds setup instructions, a clearer project map, and
reproducibility notes without changing the experimental logic.

## Repository Status

- The project is organized around Jupyter notebooks rather than command-line
  training scripts.
- The repository includes prepared data artifacts in `data/`.
- Some notebooks still contain environment-specific paths from the original
  working setup, including Kaggle paths and local Windows paths. Update the
  `DATA_PATH`, checkpoint, and output directory variables inside the notebooks
  before running them in a new environment.
- No official experiment metrics or paper-ready results are claimed here beyond
  what is already visible in the notebooks.

## Project Structure

```text
.
+-- data/
|   +-- datamaps.json              # User/item/attribute id mappings
|   +-- heterodata_object.pt        # Prepared PyTorch/PyG data object
|   +-- meta.json.gz                # Compressed item metadata
|   `-- sequential_data.txt         # User interaction sequences
+-- data_prep_diff_emb.ipynb       # Data preparation and embedding workflow
+-- RQVAE_train_stage.ipynb        # RQ-VAE training workflow
+-- RQVAE_AntiContrastive_train.ipynb
|                                   # Anti-contrastive RQ-VAE variant
+-- GPT2_Rec_Analysis.ipynb        # GPT-style recommendation analysis
+-- requirements.txt               # Python dependencies inferred from notebooks
+-- CONTRIBUTING.md                # Lightweight development notes
`-- LICENSE                        # License status note
```

## Installation

The notebooks were authored with Python 3.12 kernels. A GPU-enabled PyTorch
setup is recommended for training, but the exact PyTorch build depends on your
CUDA/CPU environment.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you need CUDA support, install the PyTorch build that matches your system
before installing the rest of the requirements.

```bash
jupyter lab
```

## Data

The tracked `data/` directory contains the files required by the preparation
and modeling notebooks:

- `sequential_data.txt`: integer item sequences per user.
- `datamaps.json`: mappings between original ids and internal numeric ids.
- `meta.json.gz`: compressed item metadata used during feature construction.
- `heterodata_object.pt`: serialized PyTorch/PyG object for downstream models.

The training notebooks currently refer to `heterodata_object12_updated.pt` in
their original Kaggle/local paths. When running locally, either update
`DATA_PATH` to point at `data/heterodata_object.pt` if that artifact is suitable
for your run, or regenerate the updated artifact from `data_prep_diff_emb.ipynb`
and place it where the training notebooks expect it.

## Usage

Run the notebooks in the following order for the full workflow:

1. `data_prep_diff_emb.ipynb`
   - Loads the sequential data, metadata, and id maps.
   - Builds item-side representations and a heterogeneous data object.
   - Saves a prepared `.pt` artifact for model training.

2. `RQVAE_train_stage.ipynb`
   - Trains the main RQ-VAE semantic-id model.
   - Saves checkpoints named like `rqvae_improved_s{session}.pt`.

3. `RQVAE_AntiContrastive_train.ipynb`
   - Trains the anti-contrastive RQ-VAE variant.
   - Saves checkpoints named like `rqvae_anti_s{session}.pt`.

4. `GPT2_Rec_Analysis.ipynb`
   - Loads trained RQ-VAE checkpoints.
   - Assigns semantic ids to items.
   - Trains/evaluates a GPT-style sequential recommendation model.
   - Saves GPT recommendation checkpoints and plots when configured.

## Training and Evaluation

Training hyperparameters such as batch size, number of epochs, learning rate,
codebook size, number of RQ-VAE layers, and checkpoint directories are defined
inside the notebooks. Review those cells before launching a run.

Expected generated artifacts include:

- `rqvae_improved_s*.pt`
- `rqvae_anti_s*.pt`
- `gpt2rec_imp_best.pt`
- `gpt2rec_anti_best.pt`
- `gpt2rec_results.png`

These generated artifacts are ignored by `.gitignore` so that large experiment
outputs are not accidentally committed.

## Development Notes

There are no standalone unit tests in the original repository. For lightweight
checks after edits, validate that the notebooks are readable JSON and that the
documented dependency list still matches notebook imports.

```bash
jupyter nbconvert --to notebook --execute RQVAE_train_stage.ipynb
```

The command above executes a full notebook and may be slow or require a GPU and
correct local data paths. For documentation-only changes, JSON parsing of the
notebooks is usually a safer smoke check.

## Citation and References

No canonical BibTeX entry or publication metadata was included in the upstream
repository. If this project is submitted for coursework or publication, cite the
course/project repository and add formal citations for the specific generative
retrieval, RQ-VAE, transformer, dataset, and embedding-model references used in
your report.

Core libraries used by the notebooks include:

- PyTorch
- PyTorch Geometric
- Sentence-Transformers
- Gensim
- Jupyter

## License

The upstream repository did not include an explicit open-source license at the
time of this cleanup. See `LICENSE` for the current license-status note. Choose
and add a concrete license before redistributing this project as open source.
