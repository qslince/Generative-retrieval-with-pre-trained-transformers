import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / '.env')

s3 = boto3.client('s3')
bucket_name = os.environ['bucket_name']


def uploadDirectory(path, bucket, prefix=''):
    path = Path(path)
    if not path.is_dir():
        raise SystemExit(f'нет каталога {path}')
    for root, _, files in os.walk(path):
        for file in files:
            local = os.path.join(root, file)
            key = prefix + os.path.relpath(local, path).replace(os.sep, '/')
            print(local, '->', key)
            s3.upload_file(local, bucket, key)


uploadDirectory(ROOT / 'data', bucket=bucket_name, prefix='raw/')