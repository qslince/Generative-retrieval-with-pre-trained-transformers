# SID-метрики и как их интерпретировать

## Collision metrics

- `n_collisions` - сколько item'ов не имеют уникального base SID.
- `cfr` - collision-free ratio. Чем выше, тем больше item'ов имеют уникальный base SID.
- `max_dupe` - максимальный размер collision cluster.

Интерпретация: меньше коллизий обычно лучше, но слишком длинный SID может ухудшить autoregressive decoding.

## Entropy metrics

- `entropy_l0...entropy_lN` - насколько равномерно используются коды на каждом уровне.
- `entropy_mean` - средняя энтропия уровней.
- `entropy_min` - слабейший уровень. Если один уровень коллапсирует, он может ограничивать весь SID.

Интерпретация: высокая entropy полезна, если она не является случайным шумом.

## Similarity metrics

- `pas_emb` - средняя cosine similarity item embeddings внутри collision clusters.
- `pas_behavioral` - средняя похожесть по множествам пользователей.

Интерпретация: если collision items реально похожи, коллизия менее вредна для RecSys.

## Distribution metrics

- `zipf_alpha_full` - форма распределения частот SID.
- `cur_total` - доля использованного пространства кодов.

Интерпретация: слишком сильный Zipf/collapse обычно вреден, но `cur_total` сам по себе может быть малоинформативен при огромном пространстве кодов.

## Связь с RecSys

Считать:

- Spearman между каждой SID-метрикой и `Recall@K`/`NDCG@K`;
- корреляции между самими SID-метриками;
- отдельно общий анализ и controlled ablation.

В выводах не писать "корреляция доказывает причинность". Правильная формулировка: "метрика является прокси-кандидатом и требует controlled ablation".
