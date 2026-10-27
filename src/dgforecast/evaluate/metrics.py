"""The scorecard. Retail forecasting is judged mostly on WAPE and bias.

WAPE (weighted absolute percentage error) = total absolute error / total actual units.
  It's percentage error weighted by volume, so a slow item that sells 1 unit
  can't blow up the score the way plain MAPE does.
Bias = (total forecast - total actual) / total actual.
  Positive means we over-forecast (too much stock), negative means under (empty shelves).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / max(y.sum(), 1e-9))


def bias(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float((yhat.sum() - y.sum()) / max(y.sum(), 1e-9))


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def dollar_impact(y, yhat, price, margin_rate: float, holding_rate: float) -> dict:
    """Lost margin from under-forecasting plus holding cost from over-forecasting."""
    y, yhat, price = (np.asarray(a, float) for a in (y, yhat, price))
    under = np.clip(y - yhat, 0, None) * price * margin_rate
    over = np.clip(yhat - y, 0, None) * price * holding_rate
    return {"lost_margin_usd": float(under.sum()), "holding_cost_usd": float(over.sum()),
            "total_cost_usd": float(under.sum() + over.sum())}


def scorecard(df: pd.DataFrame, pred_col: str, by: str | None = None) -> pd.DataFrame | dict:
    def one(d):
        return {"wape": wape(d.units_sold, d[pred_col]), "bias": bias(d.units_sold, d[pred_col]),
                "mae": mae(d.units_sold, d[pred_col]), "units": float(d.units_sold.sum())}
    if by is None:
        return one(df)
    return pd.DataFrame({k: one(d) for k, d in df.groupby(by)}).T
