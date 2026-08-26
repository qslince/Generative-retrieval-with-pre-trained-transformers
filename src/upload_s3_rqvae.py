import os
import re
import sys
from pathlib import Path

import awswrangler as wr
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')

LOSS_ALIASES = {
    'anti': 'anti_contrastive',
    'improved': 'contrastive',
    'contrastive': 'contrastive',
}

METRIC_ALIASES = {'cfr': 'ICR'}

RQVAE_CFG = ['n_layers', 'codebook_size', 'embed_dim', 'beta', 'gamma',
             'ema_decay', 'temperature', 'margin', 'hidden_sizes', 'rqvae_params']

RQVAE_METRICS = ['entropy_mean', 'entropy_min', 'entropy_l0', 'entropy_l1',
                 'entropy_l2', 'entropy_l3', 'unique_total_base', 'unique_l0',
                 'unique_l1', 'unique_l2', 'unique_l3', 'n_collisions',
                 'max_dupe', 'cfr', 'pas_emb', 'pas_behavioral', 'cur_total',
                 'zipf_alpha_full']

# в merged-таблице метрики RQ-VAE названы иначе; приводим к именам из final_table,
# чтобы одна и та же метрика сравнивалась между экспериментами
LONG_METRIC_RENAMES = {
    'rq_val_entropy_mean': 'entropy_mean', 'rq_val_entropy_min': 'entropy_min',
    'rq_val_entropy_l0': 'entropy_l0', 'rq_val_entropy_l1': 'entropy_l1',
    'rq_val_entropy_l2': 'entropy_l2', 'rq_val_entropy_l3': 'entropy_l3',
    'rq_val_sids_num_l0': 'unique_l0', 'rq_val_sids_num_l1': 'unique_l1',
    'rq_val_sids_num_l2': 'unique_l2', 'rq_val_sids_num_l3': 'unique_l3',
    'sid_unique_total': 'unique_total_base', 'sid_collisions': 'n_collisions',
    'sid_icr': 'ICR', 'sid_cur_total': 'cur_total',
}

# имена, для которых точного соответствия в final_table нет — оставляем как есть
LONG_METRIC_KEEP = ['sid_max_cluster_size', 'sid_path_avg_similarity',
                    'rqvae_val_loss', 'rqvae_best_val_loss', 'rq_val_loss']

RUN_COLS = ['run_id', 'experiment', 'loss_type', 'rq_seed', 'checkpoint',
            'checkpoint_tag', 'rqvae_epoch', *RQVAE_CFG]

METRIC_COLS = ['run_id', 'experiment', 'loss_type', 'metric', 'level', 'value']


def split_level(name):
    m = re.fullmatch(r'(.+)_l(\d+)', name)
    if m:
        return METRIC_ALIASES.get(m.group(1), m.group(1)), int(m.group(2))
    return METRIC_ALIASES.get(name, name), None


def build_sweep():
    df = pd.read_csv(ROOT / 'experiments' / 'tables' / 'final_table.csv')

    df = df[['kind', 'ckpt', *RQVAE_CFG, *RQVAE_METRICS]].drop_duplicates().reset_index(drop=True)

    runs = pd.DataFrame({
        'experiment': 'sweep',
        'loss_type': df['kind'].map(LOSS_ALIASES),
        'rq_seed': df['ckpt'].str.extract(r'_s(\d+)\.pt')[0].astype('Int64'),
        'checkpoint': df['ckpt'],
    })
    runs['run_id'] = 'sweep:' + runs['loss_type'] + ':s' + runs['rq_seed'].astype(str)
    runs['checkpoint_tag'] = pd.NA
    runs['rqvae_epoch'] = pd.Series([pd.NA] * len(df), dtype='Int64')
    runs = runs.join(df[RQVAE_CFG])

    metrics = (df[RQVAE_METRICS]
               .join(runs[['run_id', 'experiment', 'loss_type']])
               .melt(id_vars=['run_id', 'experiment', 'loss_type'],
                     var_name='name', value_name='value'))
    return runs, metrics


def build_paired():
    df = pd.read_csv(ROOT / 'results' / 'tables' / 'table_rqvae_paired_by_seed.csv')

    long = df.melt(id_vars=['rq_seed'], var_name='column', value_name='value')
    parts = long['column'].str.extract(
        r'^(?P<name>ICR|CollisionRate)_(?P<arm>anti|contrastive|delta)$')
    long = long.join(parts)
    long = long[long['arm'].isin(['anti', 'contrastive'])].copy()

    long['loss_type'] = long['arm'].map(LOSS_ALIASES)
    long['experiment'] = 'paired'
    long['run_id'] = ('paired:' + long['loss_type'] + ':rq' + long['rq_seed'].astype(str))

    runs = (long[['run_id', 'experiment', 'loss_type', 'rq_seed']]
            .drop_duplicates()
            .astype({'rq_seed': 'Int64'})
            .reset_index(drop=True))
    for col in ['checkpoint', 'checkpoint_tag', 'rqvae_epoch', *RQVAE_CFG]:
        runs[col] = pd.NA

    metrics = long[['run_id', 'experiment', 'loss_type', 'name', 'value']]
    return runs, metrics


def build_longitudinal():
    df = pd.read_csv(ROOT / 'experiments' / 'tables' / 'checkpoint_sid_recsys_merged.csv')

    df = df.assign(checkpoint='l4_seed' + df['rqvae_seed'].astype(str) + '_'
                              + df['checkpoint_tag'] + '_ep' + df['rqvae_epoch'].astype(str))
    # 159 строк merged = прогоны GPT2Rec; токенизаторов среди них 18
    df = df.drop_duplicates('checkpoint').reset_index(drop=True)

    runs = pd.DataFrame({
        'experiment': 'longitudinal',
        'loss_type': pd.NA,
        'rq_seed': df['rqvae_seed'].astype('Int64'),
        'checkpoint': df['checkpoint'],
        'checkpoint_tag': df['checkpoint_tag'],
        'rqvae_epoch': df['rqvae_epoch'].astype('Int64'),
    })
    runs['run_id'] = 'longitudinal:' + df['checkpoint']
    for col in RQVAE_CFG:
        runs[col] = df[col] if col in df.columns else pd.NA

    value_cols = [*LONG_METRIC_RENAMES, *LONG_METRIC_KEEP]
    metrics = (df[value_cols]
               .rename(columns=LONG_METRIC_RENAMES)
               .join(runs[['run_id', 'experiment', 'loss_type']])
               .melt(id_vars=['run_id', 'experiment', 'loss_type'],
                     var_name='name', value_name='value'))
    return runs, metrics


def main(dry_run):
    sweep_runs, sweep_metrics = build_sweep()
    paired_runs, paired_metrics = build_paired()
    long_runs, long_metrics = build_longitudinal()

    runs = pd.concat([sweep_runs, paired_runs, long_runs], ignore_index=True)[RUN_COLS]

    metrics = pd.concat([sweep_metrics, paired_metrics, long_metrics], ignore_index=True)
    metrics[['metric', 'level']] = metrics['name'].apply(
        lambda n: pd.Series(split_level(n), index=['metric', 'level']))
    metrics['level'] = metrics['level'].astype('Int64')
    metrics = metrics[METRIC_COLS]

    assert runs['run_id'].is_unique, 'run_id не уникален'

    print(f'rqvae_runs:    {len(runs)} строк ('
          + ' / '.join(f'{n} {e}' for e, n in runs['experiment'].value_counts().items()) + ')')
    print(f'rqvae_metrics: {len(metrics)} строк, метрики: '
          f'{sorted(metrics["metric"].unique())}')

    if dry_run:
        print('\n--- rqvae_runs ---')
        print(runs.head(3).to_string())
        print('\n--- rqvae_metrics ---')
        print(metrics.head(3).to_string())
        print('\n--dry-run: в S3 ничего не записано')
        return

    wr.catalog.create_database(name='genret', exist_ok=True)

    for table, df in [('rqvae_runs', runs), ('rqvae_metrics', metrics)]:
        wr.catalog.delete_table_if_exists(database='genret', table=table)
        wr.s3.to_parquet(
            df=df,
            path=f's3://{os.environ["bucket_name"]}/tables/{table}/',
            dataset=True,
            database='genret',
            table=table,
            partition_cols=['experiment'],
            mode='overwrite',
        )
        print(f'записано: {table}')


if __name__ == '__main__':
    main(dry_run='--dry-run' in sys.argv)
