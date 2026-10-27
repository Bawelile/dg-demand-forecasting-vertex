"""Lesson 4. Turn raw sales into model features, without leaking the future.

The leakage rule: we forecast up to 14 days ahead, so on the day we make the forecast we
only know sales up to "today". A feature for a target day t may therefore only use sales
from day t-14 or earlier. That's why every sales feature here starts at lag 14 and why
there's no "yesterday's sales" feature. Prices, promotions, and the calendar are planned
in advance, so they can be used for the target day itself.

Stockouts: on a stockout day, recorded sales understate demand. We blank those values
(NaN) before computing lags and averages, and we don't train on them. XGBoost handles
the blanks natively.

The same logic exists in SQL for BigQuery in features/sql/features.sql.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from dgforecast.config import get_paths, load_config
from dgforecast.data.generate import CATEGORIES, FORMATS, REGIONS

KEYS = ["date", "store_id", "item_id"]
TARGET = "units_sold"
FEATURES = [
    # recent demand, always at least 14 days back
    "lag_14", "lag_21", "lag_28", "lag_364",
    "rmean_7", "rmean_28", "rmean_56", "rstd_28",
    # the same item across all stores, and the same store across all items (helps new stores)
    "item_chain_rmean_28", "store_total_rmean_28",
    # price and promotion (planned in advance)
    "price", "price_index", "on_promo",
    # calendar (known in advance)
    "dow", "dom", "month", "is_snap_window", "is_paycheck_friday", "is_holiday", "days_to_holiday",
    # store and item attributes
    "store_format", "region", "category", "store_age_days",
]


def build_features(sales: pd.DataFrame, stores: pd.DataFrame, items: pd.DataFrame,
                   calendar: pd.DataFrame, cfg: dict, plan: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one row per store-item-day with FEATURES (and the target when known).

    If `plan` (future days with prices/promos but no sales) is given, it's appended to
    history and only the plan rows are returned: those are the rows we forecast.
    """
    min_lag = cfg["forecast"]["min_lag_days"]
    df = sales.copy()
    if plan is not None:
        p = plan.copy(); p["units_sold"] = np.nan; p["stockout_flag"] = 0; p["_is_plan"] = 1
        df["_is_plan"] = 0
        df = pd.concat([df, p], ignore_index=True)
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    clean = df["units_sold"].where(df["stockout_flag"] == 0)  # blank censored days
    g = clean.groupby([df.store_id, df.item_id])
    for k in (min_lag, min_lag + 7, min_lag + 14, 364):
        df[f"lag_{k}"] = g.shift(k)
    df = df.rename(columns={f"lag_{min_lag}": "lag_14", f"lag_{min_lag + 7}": "lag_21",
                            f"lag_{min_lag + 14}": "lag_28"})
    base = g.shift(min_lag)  # everything below is computed on sales >= 14 days old
    bg = base.groupby([df.store_id, df.item_id])
    for w in (7, 28, 56):
        df[f"rmean_{w}"] = bg.transform(lambda s, w=w: s.rolling(w, min_periods=max(3, w // 4)).mean())
    df["rstd_28"] = bg.transform(lambda s: s.rolling(28, min_periods=7).std())
    df["item_chain_rmean_28"] = df.groupby(["date", "item_id"]).rmean_28.transform("mean")
    df["store_total_rmean_28"] = df.groupby(["date", "store_id"]).rmean_28.transform("sum")

    df = df.merge(calendar[["date", "dow", "dom", "month", "is_snap_window", "is_paycheck_friday",
                            "is_holiday", "days_to_holiday"]], on="date", how="left")
    st = stores.assign(store_format=stores["format"].map({f: i for i, f in enumerate(FORMATS)}),
                       region_code=stores["region"].map({r: i for i, r in enumerate(REGIONS)}))
    df = df.merge(st[["store_id", "store_format", "region_code", "open_date"]], on="store_id", how="left")
    df = df.rename(columns={"region_code": "region"})
    it = items.assign(category_code=items["category"].map({c: i for i, c in enumerate(CATEGORIES)}))
    df = df.merge(it[["item_id", "category_code", "base_price"]], on="item_id", how="left")
    df = df.rename(columns={"category_code": "category"})
    df["price_index"] = df["price"] / df["base_price"]
    df["store_age_days"] = (df["date"] - pd.to_datetime(df["open_date"])).dt.days

    if plan is not None:
        df = df[df["_is_plan"] == 1]          # forecast rows only
    else:
        df = df[df["rmean_28"].notna()]       # drop each series' first weeks (no history yet)
    cols = KEYS + FEATURES + [TARGET, "stockout_flag"]
    return df[cols].reset_index(drop=True)


def load_raw(paths):
    r = paths.raw
    return (pd.read_parquet(r / "sales_history.parquet"), pd.read_parquet(r / "stores.parquet"),
            pd.read_parquet(r / "items.parquet"), pd.read_parquet(r / "calendar.parquet"))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--out")
    a = ap.parse_args(argv)
    cfg = load_config(a.config); paths = get_paths(cfg, a.out)
    sales, stores, items, cal = load_raw(paths)
    feats = build_features(sales, stores, items, cal, cfg)
    feats.to_parquet(paths.features / "features.parquet", index=False)
    print(f"features: {len(feats):,} rows x {len(FEATURES)} features")


if __name__ == "__main__":
    main()
