"""Lesson 7. Nightly batch prediction: forecast the next 14 days for every store and item.

Because every sales feature is at least 14 days old, we can build features for all 14
future days today, from history plus the price/promo plan. The champion model scores
them, and we write one forecast table, the same shape a replenishment system would read
from BigQuery.

Each forecast also carries its top 3 reasons (from SHAP values), so a district manager can
see why a number is high or low: Dollar General's "glass box, not black box" standard.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import xgboost as xgb

from dgforecast.config import get_paths, load_config
from dgforecast.features.build import FEATURES, build_features, load_raw
from dgforecast.registry.registry import load_champion_model

READABLE = {
    "lag_14": "sales 2 weeks ago", "lag_21": "sales 3 weeks ago", "lag_28": "sales 4 weeks ago",
    "lag_364": "sales a year ago", "rmean_7": "recent weekly level", "rmean_28": "4-week average",
    "rmean_56": "8-week average", "rstd_28": "recent volatility", "item_chain_rmean_28": "item's chain-wide demand",
    "store_total_rmean_28": "store's overall traffic", "price": "price", "price_index": "price vs normal",
    "on_promo": "promotion", "dow": "day of week", "dom": "day of month", "month": "month",
    "is_snap_window": "SNAP / first-of-month window", "is_paycheck_friday": "paycheck Friday",
    "is_holiday": "holiday", "days_to_holiday": "days to next holiday", "store_format": "store format",
    "region": "region", "category": "category", "store_age_days": "store age",
}


def top_drivers(booster: xgb.Booster, X: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Exact per-row SHAP values from XGBoost (pred_contribs), turned into plain-English reasons."""
    contrib = booster.predict(xgb.DMatrix(X, feature_names=FEATURES), pred_contribs=True)[:, :-1]
    order = np.argsort(-np.abs(contrib), axis=1)[:, :k]
    out = {}
    for j in range(k):
        idx = order[:, j]
        val = contrib[np.arange(len(X)), idx]
        out[f"driver_{j + 1}"] = [f"{READABLE[FEATURES[i]]} ({'+' if v > 0 else '-'})" for i, v in zip(idx, val)]
    return pd.DataFrame(out, index=X.index)


def batch_predict(cfg: dict, paths) -> pd.DataFrame:
    sales, stores, items, cal = load_raw(paths)
    plan = pd.read_parquet(paths.raw / "future_plan.parquet")
    origin = sales.date.max()
    horizon = cfg["forecast"]["horizon_days"]
    plan = plan[(plan.date > origin) & (plan.date <= origin + pd.Timedelta(days=horizon))]
    feats = build_features(sales, stores, items, cal, cfg, plan=plan)
    booster, ch = load_champion_model(paths)
    if booster is None:
        raise RuntimeError("No champion model yet. Run training and evaluation first.")
    feats["predicted_units"] = np.clip(booster.predict(xgb.DMatrix(feats[FEATURES], feature_names=FEATURES)), 0, None)
    drivers = top_drivers(booster, feats[FEATURES])
    out = pd.concat([feats[["date", "store_id", "item_id", "predicted_units"]], drivers], axis=1)
    out = out.rename(columns={"date": "target_date"})
    out.insert(0, "forecast_origin", origin)
    out["model_version"] = ch["version"]
    return out, feats


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--out")
    a = ap.parse_args(argv)
    cfg = load_config(a.config); paths = get_paths(cfg, a.out)
    fc, feats = batch_predict(cfg, paths)
    tag = fc.forecast_origin.iloc[0].strftime("%Y%m%d")
    fc.to_parquet(paths.predictions / f"forecast_{tag}.parquet", index=False)
    feats.to_parquet(paths.predictions / f"scoring_features_{tag}.parquet", index=False)
    print(f"forecast from {fc.forecast_origin.iloc[0].date()}: {len(fc):,} store-item-days, "
          f"{fc.predicted_units.sum():,.0f} units, model {fc.model_version.iloc[0]}")
    print(fc.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
