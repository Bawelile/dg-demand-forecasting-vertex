"""Build the feature table inside BigQuery with features/sql/features.sql.

    python -m dgforecast.cloud.bq_features --project P --dataset dg_forecast
"""
from __future__ import annotations

import argparse
from pathlib import Path

SQL = Path(__file__).resolve().parents[1] / "features" / "sql" / "features.sql"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True); ap.add_argument("--dataset", default="dg_forecast")
    a = ap.parse_args(argv)
    from google.cloud import bigquery
    sql = SQL.read_text().replace("PROJECT.DATASET", f"{a.project}.{a.dataset}")
    job = bigquery.Client(project=a.project).query(sql)
    job.result()
    print(f"built {a.dataset}.features; bytes processed: {job.total_bytes_processed:,}")


if __name__ == "__main__":
    main()
