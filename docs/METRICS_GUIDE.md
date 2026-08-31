# SID metrics and how to interpret them

## Collision metrics

- `n_collisions` — how many items lack a unique base SID.
- `cfr` — collision-free ratio. The higher it is, the more items have a unique base SID.
- `max_dupe` — the size of the largest collision cluster.

Interpretation: fewer collisions is usually better, but an overly long SID can degrade autoregressive decoding.

## Entropy metrics

- `entropy_l0...entropy_lN` — how uniformly the codes are used at each level.
- `entropy_mean` — mean entropy across levels.
- `entropy_min` — the weakest level. If one level collapses, it can bottleneck the whole SID.

Interpretation: high entropy is useful as long as it is not just random noise.

## Similarity metrics

- `pas_emb` — mean cosine similarity of item embeddings within collision clusters.
- `pas_behavioral` — mean similarity in terms of user sets.

Interpretation: if colliding items really are similar, the collision is less harmful for RecSys.

## Distribution metrics

- `zipf_alpha_full` — the shape of the SID frequency distribution.
- `cur_total` — the fraction of the code space actually used.

Interpretation: an overly steep Zipf / collapse is usually harmful, but `cur_total` on its own can be uninformative when the code space is huge.

## Relation to RecSys

Compute:

- Spearman correlation between each SID metric and `Recall@K` / `NDCG@K`;
- correlations among the SID metrics themselves;
- the overall analysis and the controlled ablation separately.

Do not write "correlation proves causation" in the conclusions. The correct phrasing is: "the metric is a candidate proxy and requires a controlled ablation."