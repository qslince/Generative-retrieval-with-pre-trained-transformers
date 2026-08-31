# Experiment protocol

## Experiments

| experiment_name | SID idea | What it tests |
|---|---|---|
| `baseline_3code_count` | original RQ-VAE + count suffix | baseline reference point |
| `improved_4code_count_suffix` | 4 RQ codes + ordinal suffix | current strong baseline |
| `improved_4code_no_suffix_popular` | 4 base codes only, collisions resolved by popularity | is a 5th code needed at all |
| `improved_4code_popularity_suffix` | the most popular item in a collision cluster gets suffix 0 | does popularity routing help |
| `improved_4code_centroid_suffix` | suffix 0 goes to the item nearest the centroid | does prototype routing help |
| `improved_4code_semantic_suffix` | suffix derived from residual k-means | does the 5th code become semantic |

## Final RecSys metrics

- `Recall@1`, `Recall@5`, `Recall@10`, `Recall@20`;
- `NDCG@1`, `NDCG@5`, `NDCG@10`, `NDCG@20`;
