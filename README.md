# Generative Retrieval with Pre-trained Transformers

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Jupyter](https://img.shields.io/badge/interface-Jupyter-orange.svg)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c.svg)
![License](https://img.shields.io/badge/license-not%20specified-lightgrey.svg)

This repository contains a research project for generative
retrieval in sequential recommendation. The workflow builds item
representations, learns residual-quantized semantic identifiers with RQ-VAE
variants, and uses a GPT-style sequence model for recommendation analysis.

## Project Structure

```text
.
+-- data/
|   +-- datamaps.json              # User/item/attribute id mappings
|   +-- heterodata_object.pt        # Prepared PyTorch data object
|   +-- meta.json.gz                # Compressed item metadata
|   `-- sequential_data.txt         # User interaction sequences
+-- notebooks/
|   `-- coursework/                # Coursework notebook pipeline
|       +-- 00_baseline/           # Data preparation and baseline notebooks
|       +-- 01_sid_generation/     # RQ-VAE/SID generation notebooks
|       +-- 02_recsys/             # GPT2Rec training/evaluation notebooks
|       `-- 03_analysis/           # Experiment analysis notebooks
+-- src/
|   `-- sid_utils.py               # Reusable SID assignment/metric helpers
+-- docs/                          # Protocol, metric guide, file map
+-- experiments/
|   +-- configs/                   # Experiment configuration notes
|   +-- figures/                   # Coursework experiment figures
|   `-- tables/                    # Coursework experiment CSV outputs
+-- results/
|   +-- figures/                   # Report figures
|   `-- tables/                    # Final CSV summaries and paired tests
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

Open notebooks from the repository root so that relative paths such as `data/`
resolve consistently.

If your Jupyter kernel starts inside `notebooks/`, run `%cd ..` in a setup cell
or adjust the data paths before executing the workflow.

Some coursework notebooks import `sid_utils`; if needed, add `src/` to
`PYTHONPATH` or run them from the repository root with `sys.path.append("src")`.

## Data

The tracked `data/` directory contains the files required by the preparation
and modeling notebooks:

- `sequential_data.txt`: integer item sequences per user.
- `datamaps.json`: mappings between original ids and internal numeric ids.
- `meta.json.gz`: item metadata used during feature construction.
- `heterodata_object.pt`: serialized PyTorch/PyG object for downstream models.

## Workflow

All notebooks are kept under `notebooks/coursework/`. The tree combines the upstream
notebooks and the expanded coursework experiments.

1. `notebooks/coursework/00_baseline/`
   - Data preparation and baseline notebooks.
   - Includes the upstream data-preparation notebook as
     `data_prep_diff_emb_upstream.ipynb`.
2. `notebooks/coursework/01_sid_generation/`
   - Improved RQ-VAE training, 3-level/4-level SID variants, longitudinal
     checkpoints, and collision disambiguation notebooks.
   - Includes the upstream RQ-VAE training notebooks as
     `RQVAE_train_stage_upstream.ipynb` and
     `RQVAE_AntiContrastive_train_upstream.ipynb`.
3. `notebooks/coursework/02_recsys/`
   - GPT2Rec training and evaluation over generated SID variants.
4. `notebooks/coursework/03_analysis/`
   - Tie-break analysis, multi-seed summaries, Spearman correlations, and
     final report tables.

## Results Summary

The `results/` directory contains compact artifacts prepared from the local runs.

### RQ-VAE Semantic Identifier Quality

The paired multi-seed comparison used 10 RQ-VAE seeds. Higher Independent Code
Rate and lower Collision Rate indicate cleaner semantic identifiers.

| RQ-VAE variant | Independent Code Rate | Collision Rate | Unique SIDs | Mean code entropy |
| --- | ---: | ---: | ---: | ---: |
| Contrastive | 0.962 +/- 0.009 | 0.038 +/- 0.009 | 11643.7 +/- 103.8 | 7.659 +/- 0.104 |
| Anti-contrastive | 0.712 +/- 0.089 | 0.288 +/- 0.089 | 8612.8 +/- 1076.1 | 6.175 +/- 0.200 |

The contrastive version won 10/10 paired RQ-VAE runs by Independent Code Rate.
The mean paired improvement was 0.250, with a bootstrap 95% CI of
`[0.201, 0.304]` and one-sided Wilcoxon `p=0.00098`.

Source tables:

- `results/tables/table_rqvae_summary_by_loss.csv`
- `results/tables/table_rqvae_paired_stat_test.csv`
- `experiments/tables/final_table.csv`
- `experiments/tables/tiebreak_result_l3.csv`
- `experiments/tables/tiebreak_result_l4.csv`

### GPT2Rec Downstream Quality

The downstream comparison used 30 paired runs: 10 RQ-VAE seeds times 3 GPT2Rec
seeds. The contrastive identifiers won all 30 paired runs for Recall, NDCG, and
MRR metrics tracked in the report tables.

| RQ-VAE identifiers | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contrastive | 0.01303 +/- 0.00255 | 0.02069 +/- 0.00369 | 0.00650 +/- 0.00151 | 0.00843 +/- 0.00177 | 0.00563 +/- 0.00134 |
| Anti-contrastive | 0.00247 +/- 0.00129 | 0.00429 +/- 0.00149 | 0.00134 +/- 0.00093 | 0.00180 +/- 0.00088 | 0.00149 +/- 0.00085 |

For NDCG@10, the mean paired improvement was 0.00517 with bootstrap 95% CI
`[0.00451, 0.00584]`; one-sided Wilcoxon `p=9.31e-10`.

Source tables:

- `results/tables/table_gpt2_summary_by_loss.csv`
- `results/tables/table_gpt2_paired_stat_tests.csv`
- `results/tables/table_gpt2_paired_by_seed.csv`

### Additional Report Findings

- For 3-level RQ-VAE identifiers, an additional semantic tie-breaking code
  improved Recall@20 from 0.07858 to 0.08352 in the report run.
- For 4-level RQ-VAE identifiers, the best reported Recall@20 was 0.08624 with
  the base identifier, without adding a fifth code.
- In checkpoint analysis, the best reported Recall@20 reached 0.10180.
- Mean code entropy was strongly associated with recommendation quality:
  Spearman correlation was 0.831 with Recall@20 and 0.812 with NDCG@20.

![Recommendation quality across RQ-VAE checkpoints](results/figures/checkpoint_test_quality_by_checkpoint.png)

![Identifier/recommendation correlation heatmap](results/figures/checkpoint_sid_recsys_corr_heatmap.png)

The broader, less-curated experiment export is kept in `experiments/`. Use
`results/` for the compact report-ready view and `experiments/` when you need
the full coursework trace.

## Experiment Warehouse (S3 + Athena)

Experiment results are also published as a small relational warehouse: Parquet
datasets in S3, registered in the AWS Glue catalog under the `genret` database
and queried with Athena. The motivation is practical — the flat CSV exports put
hyperparameters, tokenizer metrics and recommendation metrics in the same row,
so every new comparison meant another pandas script. Normalized, a new
comparison is a new query.

Four tables, each partitioned by `experiment`:

| Table | Rows | Grain |
| --- | --- | --- |
| `rqvae_runs` | 44 | one RQ-VAE run, with its configuration |
| `rqvae_metrics` | 490 | long format: run, metric, level, value |
| `gpt2_runs` | 230 | one GPT2Rec run, with its configuration |
| `gpt2_metrics` | 3222 | long format: run, split, metric, k, value |

The `experiment` partition exists because the three source families have
different units of observation and must not be averaged together:

- `sweep` — checkpoint and tie-break sweep, one GPT2Rec run per configuration;
  the GPT2 seed is not varied, so `gpt_seed` is NULL.
- `paired` — paired contrastive / anti-contrastive comparison, 10 RQ-VAE seeds
  by 3 GPT2 seeds.
- `longitudinal` — 18 RQ-VAE training checkpoints by 3 tie-break variants by
  3 GPT2 seeds. This family has no loss arm, so `loss_type` is NULL.

Metrics are stored long rather than wide because the families do not report the
same metric sets — `@1` appears only in the sweep, `MRR` only in the paired
runs — and as columns those would be NULL by construction.

### Verified against the original analysis

`src/verify_sql.py` recomputes three published results in Athena and asserts
them against the original pandas figures:

| Result | Value |
| --- | --- |
| Paired ICR gain, contrastive over anti-contrastive | 0.250466904 |
| Mean valid NDCG@10, paired, anti / contrastive | 0.001337876 / 0.006503108 |
| Spearman(mean code entropy, test Recall@20), n = 53 | 0.830915026 |

All three agree to nine decimal places. The Spearman query is the one that
justifies the schema: it joins `rqvae_metrics` to `gpt2_metrics` through the
checkpoint, averages over the three GPT2 seeds, and computes rank correlation
with window functions — `rank()` plus a tie-size correction, because Athena
returns minimum ranks where Spearman needs average ranks.

```bash
python src/upload_s3_rqvae.py --dry-run   # print the schema, write nothing
python src/upload_s3_rqvae.py             # publish both RQ-VAE tables
python src/upload_s3_gpt2.py              # publish both GPT2Rec tables
python src/verify_sql.py                  # SQL vs pandas, non-zero exit on drift
```

The bucket name is read from `bucket_name` in a local `.env` file, which is
gitignored.

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

