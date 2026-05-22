# Протокол экспериментов

## Fixed settings

Для честного сравнения во всех SID-ablation фиксируются:

- dataset и train/valid/test split;
- item embeddings;
- RQ-VAE checkpoint, если сравнивается только post-processing SID;
- архитектура GPT2Rec;
- `MAX_HIST_LEN`;
- `BATCH_SIZE`;
- learning rate schedule;
- `BEAM_SIZE`;
- random seed.

## Основные эксперименты

| experiment_name | SID idea | Что проверяет |
|---|---|---|
| `baseline_3code_count` | исходный PLUM/RQ-VAE + count suffix | базовая точка сравнения |
| `improved_4code_count_suffix` | 4 RQ-кода + порядковый suffix | текущий сильный baseline |
| `improved_4code_no_suffix_popular` | только 4 base-кода, коллизии раскрываются популярностью | нужен ли 5-й код |
| `improved_4code_popularity_suffix` | самый популярный item в collision cluster получает suffix 0 | помогает ли popularity routing |
| `improved_4code_centroid_suffix` | suffix 0 у item, ближайшего к центроиду | помогает ли prototype routing |
| `improved_4code_semantic_suffix` | suffix из residual k-means | становится ли 5-й код семантическим |

## Итоговые RecSys-метрики

- `Recall@1`, `Recall@5`, `Recall@10`, `Recall@20`;
- `NDCG@1`, `NDCG@5`, `NDCG@10`, `NDCG@20`;
- `val_loss_best`.

## Decoder diagnostics

Желательно добавить:

- `base_sid_hit@K`: попал ли правильный base SID в top-K до раскрытия suffix;
- `suffix_acc_given_base`: точность suffix, если base SID угадан;
- `collision_target_rate`: доля target item'ов из collision clusters;
- `oracle_collision_recall`: потолок качества, если base SID угадан, а item внутри cluster выбран oracle-способом.
