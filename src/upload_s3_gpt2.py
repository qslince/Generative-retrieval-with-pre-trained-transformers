import os
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

GPT2_HPARAMS = ['gpt_d_model', 'gpt_n_heads', 'gpt_n_layers', 'gpt_dropout',
                'gpt_batch_size', 'gpt_lr', 'gpt_warmup_steps', 'gpt_epochs',
                'beam_size', 'max_hist_len', 'gpt_vocab']

RUN_COLS = ['run_id', 'experiment', 'loss_type', 'rq_seed', 'gpt_seed',
            'checkpoint', 'tie_break', 'rec_source', *GPT2_HPARAMS]

METRIC_COLS = ['run_id', 'experiment', 'loss_type', 'split', 'metric', 'k', 'value']


def split_metric(name):
    metric, _, k = name.partition('@')
    return metric, (int(k) if k else None)


def build_sweep():
    df = pd.read_csv(ROOT / 'experiments' / 'tables' / 'final_table.csv')

    runs = pd.DataFrame({
        'experiment': 'sweep',
        'loss_type': df['kind'].map(LOSS_ALIASES),
        'rq_seed': df['ckpt'].str.extract(r'_s(\d+)\.pt')[0].astype('Int64'),
        'gpt_seed': pd.Series([pd.NA] * len(df), dtype='Int64'),
        'checkpoint': df['ckpt'],
        'tie_break': df['tie_break'],
        'rec_source': df['rec_source'],
    })
    runs['run_id'] = ('sweep:' + runs['loss_type'] + ':s' + runs['rq_seed'].astype(str)
                      + ':' + runs['tie_break'])
    runs = runs.join(df[GPT2_HPARAMS])

    value_cols = [c for c in df.columns
                  if c.startswith(('Recall@', 'NDCG@')) or c == 'val_loss_best']
    metrics = (df[value_cols]
               .join(runs[['run_id', 'experiment', 'loss_type']])
               .melt(id_vars=['run_id', 'experiment', 'loss_type'],
                     var_name='name', value_name='value'))
    metrics['split'] = pd.NA
    return runs, metrics


def build_paired():
    df = pd.read_csv(ROOT / 'results' / 'tables' / 'table_gpt2_paired_by_seed.csv')

    long = df.melt(id_vars=['rq_seed', 'gpt_seed'], var_name='column', value_name='value')
    parts = long['column'].str.extract(r'^valid_(?P<name>.+)_(?P<arm>anti|contrastive|delta)$')
    long = long.join(parts)
    long = long[long['arm'] != 'delta'].copy()

    long['loss_type'] = long['arm'].map(LOSS_ALIASES)
    long['experiment'] = 'paired'
    long['split'] = 'valid'
    long['run_id'] = ('paired:' + long['loss_type']
                      + ':rq' + long['rq_seed'].astype(str)
                      + ':gpt' + long['gpt_seed'].astype(str))

    runs = (long[['run_id', 'experiment', 'loss_type', 'rq_seed', 'gpt_seed']]
            .drop_duplicates()
            .astype({'rq_seed': 'Int64', 'gpt_seed': 'Int64'})
            .reset_index(drop=True))
    for col in ['checkpoint', 'tie_break', 'rec_source', *GPT2_HPARAMS]:
        runs[col] = pd.NA

    metrics = long[['run_id', 'experiment', 'loss_type', 'split', 'name', 'value']]
    return runs, metrics


def build_longitudinal():
    df = pd.read_csv(ROOT / 'experiments' / 'tables' / 'checkpoint_sid_recsys_merged.csv')
    df = df.rename(columns={'gpt2_best_val_loss': 'val_loss_best'})

    checkpoint = ('l4_seed' + df['rqvae_seed'].astype(str) + '_' + df['checkpoint_tag']
                  + '_ep' + df['rqvae_epoch'].astype(str))

    runs = pd.DataFrame({
        'experiment': 'longitudinal',
        'loss_type': pd.NA,
        'rq_seed': df['rqvae_seed'].astype('Int64'),
        'gpt_seed': df['gpt2_seed'].astype('Int64'),
        'checkpoint': checkpoint,
        'tie_break': df['tiebreak'],
        'rec_source': pd.NA,
    })
    runs['run_id'] = ('longitudinal:' + checkpoint + ':' + df['tiebreak']
                      + ':gpt' + df['gpt2_seed'].astype(str))
    for col in GPT2_HPARAMS:
        runs[col] = pd.NA

    value_cols = [c for c in df.columns
                  if c.startswith(('val_Recall@', 'val_NDCG@', 'test_Recall@', 'test_NDCG@'))]
    value_cols.append('val_loss_best')

    metrics = (df[value_cols]
               .join(runs[['run_id', 'experiment', 'loss_type']])
               .melt(id_vars=['run_id', 'experiment', 'loss_type'],
                     var_name='name', value_name='value'))
    parts = metrics['name'].str.extract(r'^(val|test)_((?:Recall|NDCG)@\d+)$')
    metrics['split'] = parts[0]
    metrics['name'] = parts[1].where(parts[1].notna(), metrics['name'])
    return runs, metrics


def main(dry_run):
    sweep_runs, sweep_metrics = build_sweep()
    paired_runs, paired_metrics = build_paired()
    long_runs, long_metrics = build_longitudinal()

    runs = pd.concat([sweep_runs, paired_runs, long_runs], ignore_index=True)[RUN_COLS]

    metrics = pd.concat([sweep_metrics, paired_metrics, long_metrics], ignore_index=True)
    metrics[['metric', 'k']] = metrics['name'].apply(
        lambda n: pd.Series(split_metric(n), index=['metric', 'k']))
    metrics['k'] = metrics['k'].astype('Int64')
    metrics = metrics[METRIC_COLS]

    assert runs['run_id'].is_unique, 'run_id is not unique'

    print(f'gpt2_runs:{len(runs)}  ('
          + ' / '.join(f'{n} {e}' for e, n in runs['experiment'].value_counts().items()) + ')')
    print(f'gpt2_metrics: {len(metrics)}, metrics: '
          f'{sorted(metrics["metric"].unique())}')

    if dry_run:
        print('\n--- gpt2_runs ---')
        print(runs.head(3).to_string())
        print('\n--- gpt2_metrics ---')
        print(metrics.head(3).to_string())
        print('\n--dry-run: в S3 ничего не записано')
        return

    wr.catalog.create_database(name='genret', exist_ok=True)

    for table, df in [('gpt2_runs', runs), ('gpt2_metrics', metrics)]:
        wr.s3.to_parquet(
            df=df,
            path=f's3://{os.environ["bucket_name"]}/tables/{table}/',
            dataset=True,
            database='genret',
            table=table,
            partition_cols=['experiment'],
            mode='overwrite',
        )
        print(f'Written: {table}')


if __name__ == '__main__':
    main(dry_run='--dry-run' in sys.argv)
