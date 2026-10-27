"""Lesson 5. Train an XGBoost demand model with a time-based split.

Why a time-based split? In real life we always predict the future from the past. A random
split would let the model peek at days that come after the ones it's tested on, and its
score would look better than it will ever be in production.

    |------------- train -------------|-- valid (28d) --|-- test (28d) --|
                                       early stopping     the final exam

Why XGBoost? It's strong on tabular data, fast, handles missing values (our blanked
stockout days), and works with SHAP for explanations.
Why the Tweedie objective? Store-item sales are counts with many zeros and a long right
tail; Tweedie is built for exactly that shape.
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone

import pandas as pd
import xgboost as xgb

from dgforecast.config import get_paths, load_config, write_json
from dgforecast.evaluate.metrics import bias, wape
from dgforecast.features.build import FEATURES, TARGET


def time_split(df: pd.DataFrame, test_days: int, valid_days: int):
    end = df.date.max()
    test_start = end - pd.Timedelta(days=test_days - 1)
    valid_start = test_start - pd.Timedelta(days=valid_days)
    train = df[df.date < valid_start]
    valid = df[(df.date >= valid_start) & (df.date < test_start)]
    test = df[df.date >= test_start]
    return train, valid, test


def trainable(df: pd.DataFrame) -> pd.DataFrame:
    """Don't learn from stockout days: their sales understate demand."""
    return df[(df.stockout_flag == 0) & df[TARGET].notna()]


def train(feats: pd.DataFrame, cfg: dict, run_id: str | None = None) -> dict:
    t = cfg["train"]
    run_id = run_id or "run-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    tr, va, te = time_split(feats, t["test_days"], t["valid_days"])
    tr_, va_ = trainable(tr), trainable(va)
    dtr = xgb.DMatrix(tr_[FEATURES], label=tr_[TARGET], feature_names=FEATURES)
    dva = xgb.DMatrix(va_[FEATURES], label=va_[TARGET], feature_names=FEATURES)
    booster = xgb.train(t["params"], dtr, num_boost_round=t["num_boost_round"],
                        evals=[(dtr, "train"), (dva, "valid")],
                        early_stopping_rounds=t["early_stopping_rounds"], verbose_eval=False)
    test = trainable(te).copy()
    test["pred"] = booster.predict(xgb.DMatrix(test[FEATURES], feature_names=FEATURES),
                                   iteration_range=(0, booster.best_iteration + 1))
    info = {
        "run_id": run_id,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_dates": [str(tr.date.min().date()), str(tr.date.max().date())],
        "valid_dates": [str(va.date.min().date()), str(va.date.max().date())],
        "test_dates": [str(te.date.min().date()), str(te.date.max().date())],
        "train_rows": int(len(tr_)), "best_iteration": int(booster.best_iteration),
        "params": t["params"], "features": FEATURES,
        "test_wape": wape(test[TARGET], test.pred), "test_bias": bias(test[TARGET], test.pred),
    }
    # Drift baseline = the most recent 4 weeks (the test window). A long baseline (e.g. 6
    # months) wrongly flags normal seasonality and gradual price creep as "drift".
    reference = te.copy()
    return {"booster": booster, "info": info, "test": test, "reference": reference}


def save_run(result: dict, paths) -> "os.PathLike":
    run_dir = paths.models / result["info"]["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    result["booster"].save_model(run_dir / "model.json")
    write_json(result["info"], run_dir / "run_info.json")
    result["test"].to_parquet(run_dir / "test_predictions.parquet", index=False)
    # Baseline for drift monitoring: the model's most recent 4 weeks of inputs.
    # (Vertex AI Model Monitoring compares serving inputs against a baseline dataset the same way.)
    result["reference"].to_parquet(run_dir / "reference_sample.parquet", index=False)
    log_experiment(result["info"], paths)
    return run_dir


def log_experiment(info: dict, paths) -> None:
    """Local stand-in for Vertex AI Experiments: one row per training run.
    On Google Cloud, set VERTEX_EXPERIMENT and the same numbers go to Vertex AI."""
    f = paths.reports / "experiments.csv"
    row = {k: info[k] for k in ("run_id", "trained_at", "train_rows", "best_iteration", "test_wape", "test_bias")}
    new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
    if os.environ.get("VERTEX_EXPERIMENT"):
        from google.cloud import aiplatform  # only needed on Google Cloud
        aiplatform.init(experiment=os.environ["VERTEX_EXPERIMENT"])
        aiplatform.start_run(info["run_id"].replace("_", "-"))
        aiplatform.log_params({k: str(v) for k, v in info["params"].items()})
        aiplatform.log_metrics({"test_wape": info["test_wape"], "test_bias": info["test_bias"]})
        aiplatform.end_run()


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--out")
    a = ap.parse_args(argv)
    cfg = load_config(a.config); paths = get_paths(cfg, a.out)
    feats = pd.read_parquet(paths.features / "features.parquet")
    res = train(feats, cfg)
    run_dir = save_run(res, paths)
    i = res["info"]
    print(f"{i['run_id']}: test WAPE {i['test_wape']:.1%}, bias {i['test_bias']:+.1%}, "
          f"{i['best_iteration']} trees -> {run_dir}")


if __name__ == "__main__":
    main()
