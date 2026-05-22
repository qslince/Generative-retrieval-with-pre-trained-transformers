# Results

This directory contains report-ready artifacts copied from the local coursework
result set.

## Tables

- `tables/table_rqvae_summary_by_loss.csv`: RQ-VAE identifier metrics by loss
  type.
- `tables/table_rqvae_paired_stat_test.csv`: paired statistical test for
  Independent Code Rate.
- `tables/table_rqvae_paired_by_seed.csv`: per-seed RQ-VAE paired comparison.
- `tables/table_gpt2_summary_by_loss.csv`: downstream GPT2Rec metrics by RQ-VAE
  loss type.
- `tables/table_gpt2_paired_stat_tests.csv`: paired statistical tests for
  Recall, NDCG, and MRR.
- `tables/table_gpt2_paired_by_seed.csv`: per-run GPT2Rec paired comparison.
- `tables/table_gpt2_ndcg10_mean_by_rq_seed.csv`: NDCG@10 grouped by RQ-VAE
  seed.

## Figures

- `figures/checkpoint_test_quality_by_checkpoint.png`: recommendation quality
  across saved RQ-VAE checkpoints.
- `figures/checkpoint_sid_recsys_corr_heatmap.png`: correlations between
  identifier properties and recommendation metrics.
- `figures/spearman_all_ties_heatmap.png`: Spearman correlation overview from
  the broader tie-breaking analysis.

These artifacts document the completed experimental work; generated checkpoints
and temporary outputs remain ignored by git.
