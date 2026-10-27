"""Move data between the artifacts folder and BigQuery.

    python -m dgforecast.cloud.load_to_bigquery --project P --dataset dg_forecast --out artifacts
loads the raw tables (stores, items, calendar, sales_history) so the SQL features can run.
"""
from __future__ import annotations

import argparse

import pandas as pd

from dgforecast.config import get_paths, load_config

RAW_TABLES = ["stores", "items", "calendar", "sales_history"]


def _client(project: str):
    from google.cloud import bigquery
    return bigquery.Client(project=project), bigquery


def load_raw_tables(out_root: str, project: str, dataset: str) -> None:
    client, bq = _client(project)
    paths = get_paths(load_config(), out_root)
    for t in RAW_TABLES:
        df = pd.read_parquet(paths.raw / f"{t}.parquet")
        job = client.load_table_from_dataframe(
            df, f"{project}.{dataset}.{t}",
            job_config=bq.LoadJobConfig(write_disposition="WRITE_TRUNCATE"))
        job.result()
        print(f"loaded {len(df):,} rows into {dataset}.{t}")


def load_latest_forecast(out_root: str, project: str, dataset: str) -> None:
    client, bq = _client(project)
    paths = get_paths(load_config(), out_root)
    f = max(paths.predictions.glob("forecast_*.parquet"))
    df = pd.read_parquet(f)
    client.load_table_from_dataframe(
        df, f"{project}.{dataset}.forecasts",
        job_config=bq.LoadJobConfig(write_disposition="WRITE_APPEND")).result()
    print(f"appended {len(df):,} forecast rows from {f.name} to {dataset}.forecasts")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True); ap.add_argument("--dataset", default="dg_forecast")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    load_raw_tables(a.out, a.project, a.dataset)


if __name__ == "__main__":
    main()
