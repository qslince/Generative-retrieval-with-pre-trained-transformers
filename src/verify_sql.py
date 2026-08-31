import os
from pathlib import Path

import awswrangler as wr
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')

DATABASE = 'genret'
S3_OUTPUT = f's3://{os.environ["bucket_name"]}/athena-results/'

TOL = 1e-9


def athena(sql):
    return wr.athena.read_sql_query(sql, database=DATABASE, s3_output=S3_OUTPUT)


def spearman(df, a, b):
    r = df[[a, b]].rank(method='average')
    return r[a].corr(r[b])


def check(name, got, want):
    ok = abs(got - want) < TOL
    print(f'{"OK  " if ok else "FAIL"}  {name}\n      SQL {got:.9f}  pandas {want:.9f}')
    return ok


def check_icr_delta():
    sql = athena("""
        WITH arms AS (
          SELECT r.rq_seed,
                 max(m.value) FILTER (WHERE r.loss_type = 'contrastive') AS contrastive,
                 max(m.value) FILTER (WHERE r.loss_type = 'anti_contrastive') AS anti
          FROM rqvae_runs r JOIN rqvae_metrics m ON m.run_id = r.run_id
          WHERE r.experiment = 'paired' AND m.metric = 'ICR'
          GROUP BY r.rq_seed
        )
        SELECT avg(contrastive - anti) AS mean_delta FROM arms
    """)['mean_delta'][0]

    ref = pd.read_csv(ROOT / 'results' / 'tables' / 'table_rqvae_paired_by_seed.csv')
    return check('ICR mean_delta (paired, n=10)', sql, ref['ICR_delta'].mean())


def check_gpt2_ndcg10():
    sql = athena("""
        SELECT loss_type, avg(value) AS mean_value
        FROM gpt2_metrics
        WHERE experiment = 'paired' AND metric = 'NDCG' AND k = 10
        GROUP BY loss_type ORDER BY loss_type
    """).set_index('loss_type')['mean_value']

    ref = pd.read_csv(ROOT / 'results' / 'tables' / 'table_gpt2_paired_by_seed.csv')
    pairs = {'anti_contrastive': 'valid_NDCG@10_anti',
             'contrastive': 'valid_NDCG@10_contrastive'}
    return all(check(f'NDCG@10 mean, {lt} (paired, n=30)', sql[lt], ref[col].mean())
               for lt, col in pairs.items())


def check_entropy_recall_corr():
    sql = athena("""
        WITH gpt AS (
          SELECT g.checkpoint, g.tie_break, avg(m.value) AS score
          FROM gpt2_runs g JOIN gpt2_metrics m ON m.run_id = g.run_id
          WHERE g.experiment = 'longitudinal'
            AND m.split = 'test' AND m.metric = 'Recall' AND m.k = 20
          GROUP BY g.checkpoint, g.tie_break
        ),
        ent AS (
          SELECT r.checkpoint, m.value AS entropy
          FROM rqvae_runs r JOIN rqvae_metrics m ON m.run_id = r.run_id
          WHERE r.experiment = 'longitudinal' AND m.metric = 'entropy_mean'
        ),
        pairs AS (
          SELECT g.score, e.entropy FROM gpt g JOIN ent e ON e.checkpoint = g.checkpoint
        ),
        ranked AS (
          SELECT rank() OVER (ORDER BY entropy)
                   + (count(*) OVER (PARTITION BY entropy) - 1) / 2.0 AS x,
                 rank() OVER (ORDER BY score)
                   + (count(*) OVER (PARTITION BY score) - 1) / 2.0   AS y
          FROM pairs
        )
        SELECT count(*) AS n, corr(x, y) AS spearman FROM ranked
    """)

    merged = pd.read_csv(ROOT / 'experiments' / 'tables' / 'checkpoint_sid_recsys_merged.csv')
    g = (merged.groupby(['rqvae_seed', 'checkpoint_tag', 'rqvae_epoch', 'tiebreak'], dropna=False)
         [['rq_val_entropy_mean', 'test_Recall@20']].mean().dropna())

    n_ok = int(sql['n'][0]) == len(g)
    print(f'{"OK" if n_ok else "FAIL"}  size: SQL {sql["n"][0]}  pandas {len(g)}')
    corr_ok = check('Spearman(entropy_mean, test_Recall@20) (longitudinal)',
                    sql['spearman'][0], spearman(g, 'rq_val_entropy_mean', 'test_Recall@20'))
    return n_ok and corr_ok


def main():
    checks = [check_icr_delta, check_gpt2_ndcg10, check_entropy_recall_corr]
    results = [fn() for fn in checks]
    print()
    if all(results):
        print(f'All tests were successful ({len(results)} from {len(results)})')
    else:
        raise SystemExit(f'Incorrect: {results.count(False)} из {len(results)}')


if __name__ == '__main__':
    main()
