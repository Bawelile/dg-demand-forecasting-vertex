"""Lesson 3. Check the data before anything trains on it.

A model trained on broken data still produces numbers; they're just wrong. So the
pipeline's first job is to stop itself when the data looks bad. Each check below
returns pass/fail with a reason, and the step fails loudly if any check fails.
"""
from __future__ import annotations

import argparse

import pandas as pd

from dgforecast.config import get_paths, load_config, write_json

REQUIRED = {
    "date": "datetime64[ns]", "store_id": "object", "item_id": "object", "units_sold": "int",
    "price": "float", "on_promo": "int", "stockout_flag": "int",
}


class DataValidationError(Exception):
    pass


def run_checks(sales: pd.DataFrame, cfg: dict) -> list[dict]:
    v = cfg["validation"]
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    missing = [c for c in REQUIRED if c not in sales.columns]
    add("required_columns", not missing, f"missing: {missing}" if missing else "all present")
    if missing:
        return checks

    add("min_rows", len(sales) >= v["min_rows"], f"{len(sales):,} rows (need {v['min_rows']:,})")
    null_rate = float(sales[list(REQUIRED)].isna().mean().max())
    add("null_rate", null_rate <= v["max_null_rate"], f"worst column null rate {null_rate:.3%}")
    dups = int(sales.duplicated(["date", "store_id", "item_id"]).sum())
    add("duplicates", dups <= v["max_duplicate_rows"], f"{dups} duplicate store-item-days")
    neg = int((sales.units_sold < 0).sum())
    add("non_negative_units", neg == 0, f"{neg} negative sales rows")
    bad_price = int((sales.price <= 0).sum())
    add("positive_prices", bad_price == 0, f"{bad_price} rows with price <= 0")
    flags_ok = sales.on_promo.isin([0, 1]).all() and sales.stockout_flag.isin([0, 1]).all()
    add("binary_flags", flags_ok, "on_promo and stockout_flag are 0/1" if flags_ok else "non-0/1 values")
    # every store-item series should have one row per day from its first to last date
    g = sales.groupby(["store_id", "item_id"]).date.agg(["min", "max", "count"])
    expected = (g["max"] - g["min"]).dt.days + 1
    gaps = int((expected != g["count"]).sum())
    add("no_missing_days", gaps == 0, f"{gaps} series with missing days")
    last = sales.date.max()
    add("fresh_data", True, f"latest date {last.date()}")
    return checks


def validate(sales: pd.DataFrame, cfg: dict) -> dict:
    checks = run_checks(sales, cfg)
    report = {"passed": all(c["passed"] for c in checks), "checks": checks}
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--out")
    a = ap.parse_args(argv)
    cfg = load_config(a.config); paths = get_paths(cfg, a.out)
    sales = pd.read_parquet(paths.raw / "sales_history.parquet")
    report = validate(sales, cfg)
    write_json(report, paths.reports / "validation.json")
    for c in report["checks"]:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")
    if not report["passed"]:
        raise DataValidationError("Data validation failed; stopping the pipeline.")


if __name__ == "__main__":
    main()
