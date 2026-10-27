"""Lesson 8. Explain the model: which inputs drive forecasts overall.

Uses XGBoost's built-in exact SHAP values (pred_contribs). A SHAP value says how much one
feature pushed one forecast up or down from the average. Averaging their size across many
forecasts ranks the features by importance, and the sign tells you the direction.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from dgforecast.config import get_paths, load_config, write_json
from dgforecast.features.build import FEATURES
from dgforecast.predict.batch import READABLE
from dgforecast.registry.registry import champion, load_champion_model


def global_importance(booster, X: pd.DataFrame) -> pd.Series:
    contrib = booster.predict(xgb.DMatrix(X, feature_names=FEATURES), pred_contribs=True)[:, :-1]
    return pd.Series(np.abs(contrib).mean(axis=0), index=FEATURES).sort_values(ascending=False)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--out")
    a = ap.parse_args(argv)
    cfg = load_config(a.config); paths = get_paths(cfg, a.out)
    booster, ch = load_champion_model(paths)
    run = paths.models / ch["run_id"]
    test = pd.read_parquet(run / "test_predictions.parquet")
    X = test[FEATURES].sample(min(5000, len(test)), random_state=0)
    imp = global_importance(booster, X)
    write_json({READABLE[k]: round(float(v), 4) for k, v in imp.items()}, paths.reports / "feature_importance.json")
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh([READABLE[k] for k in top.index], top.values, color="#1A6B8A")
    ax.set_xlabel("Average impact on forecast (units, mean |SHAP|)")
    ax.set_title(f"What drives the forecast (model {ch['version']})", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(paths.reports / "feature_importance.png", dpi=150)
    print("top drivers:", ", ".join(READABLE[k] for k in imp.index[:6]))


if __name__ == "__main__":
    main()
