"""Lesson 9. Monitor the model in production. Two kinds of warning:

1. Input drift (before we know the answers): are today's model inputs distributed like the
   data the model was trained on? Measured with PSI (population stability index):
       PSI < 0.10  no real change | 0.10-0.20 moderate | > 0.20 big shift: investigate
   Vertex AI Model Monitoring does this same comparison for you on Google Cloud.

2. Outcome decay (after actual sales arrive): join the forecasts to what really sold and
   compute WAPE and bias. Alert if WAPE is more than 20% worse than at training time.

Either alert can trigger retraining (pipeline/monitor_and_retrain.py).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from dgforecast.config import get_paths, load_config, write_json
from dgforecast.evaluate.metrics import bias, wape
from dgforecast.features.build import FEATURES
from dgforecast.registry.registry import champion

DRIFT_FEATURES = ["rmean_28", "lag_14", "item_chain_rmean_28", "store_total_rmean_28", "price", "price_index", "on_promo"]


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref, cur = reference.dropna().to_numpy(), current.dropna().to_numpy()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (nearly) constant feature: compare the share of each value instead
        vals = np.union1d(np.unique(ref), np.unique(cur))
        r = np.array([(ref == v).mean() for v in vals]); c = np.array([(cur == v).mean() for v in vals])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref); c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def input_drift(reference: pd.DataFrame, scoring: pd.DataFrame, cfg: dict, categories=None) -> dict:
    out = {"overall": {f: round(psi(reference[f], scoring[f]), 4) for f in DRIFT_FEATURES}}
    # Drift inside one category can hide in the overall average, so check each category too.
    # Use regular-price days only: promotions are planned price drops, not drift.
    out["price_index_by_category"] = {}
    ref_reg, sc_reg = reference[reference.on_promo == 0], scoring[scoring.on_promo == 0]
    for code, name in (categories or {}).items():
        r, s = ref_reg[ref_reg.category == code], sc_reg[sc_reg.category == code]
        if len(r) and len(s):
            out["price_index_by_category"][name] = round(psi(r.price_index, s.price_index), 4)
    lim = cfg["monitoring"]["psi_alert"]
    flagged = [f for f, v in out["overall"].items() if v > lim]
    flagged += [f"price_index[{k}]" for k, v in out["price_index_by_category"].items() if v > lim]
    out["flagged"] = flagged
    return out


def outcome_decay(forecast: pd.DataFrame, actuals: pd.DataFrame, stores: pd.DataFrame, cfg: dict, paths) -> dict:
    j = forecast.merge(actuals.rename(columns={"date": "target_date"}),
                       on=["target_date", "store_id", "item_id"], how="inner")
    j = j[j.stockout_flag == 0].merge(stores[["store_id", "region"]], on="store_id")
    ch = champion(paths)
    live = wape(j.units_sold, j.predicted_units)
    limit = ch["test_wape"] * cfg["monitoring"]["wape_alert_ratio"]
    by_region = {r: {"wape": round(wape(d.units_sold, d.predicted_units), 4),
                     "bias": round(bias(d.units_sold, d.predicted_units), 4)} for r, d in j.groupby("region")}
    # An overall average can look fine while one region is badly off, so check each region's bias.
    blim = cfg["monitoring"]["region_bias_alert"]
    regions_off = [r for r, m in by_region.items() if abs(m["bias"]) > blim]
    return {"days_scored": int(j.target_date.nunique()), "live_wape": round(live, 4),
            "live_bias": round(bias(j.units_sold, j.predicted_units), 4),
            "training_wape": round(ch["test_wape"], 4), "alert_threshold": round(limit, 4),
            "wape_alert": bool(live > limit), "regions_off": regions_off,
            "alert": bool(live > limit or regions_off), "by_region": by_region}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--out")
    ap.add_argument("--forecast-tag", required=True, help="e.g. 20251230")
    a = ap.parse_args(argv)
    cfg = load_config(a.config); paths = get_paths(cfg, a.out)
    from dgforecast.data.generate import CATEGORIES
    ch = champion(paths)
    ref = pd.read_parquet(paths.models / ch["run_id"] / "reference_sample.parquet")
    scoring = pd.read_parquet(paths.predictions / f"scoring_features_{a.forecast_tag}.parquet")
    drift = input_drift(ref, scoring, cfg, dict(enumerate(CATEGORIES)))
    fc = pd.read_parquet(paths.predictions / f"forecast_{a.forecast_tag}.parquet")
    act = pd.read_parquet(paths.raw / "future_actuals.parquet")
    stores = pd.read_parquet(paths.raw / "stores.parquet")
    decay = outcome_decay(fc, act, stores, cfg, paths)
    report = {"input_drift": drift, "outcome_decay": decay,
              "retrain": bool(drift["flagged"] or decay["alert"])}
    write_json(report, paths.reports / f"monitoring_{a.forecast_tag}.json")
    print(report)


if __name__ == "__main__":
    main()
