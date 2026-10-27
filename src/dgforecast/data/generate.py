"""Lesson 2. Make realistic, fake store-by-item daily sales for a dollar-store chain.

Why synthetic? Real Dollar General data is private. A good simulator lets you build and
test the whole system, and you control the truth, so you can check the model finds it.

What makes it realistic (each one is something the model has to learn):
  * store formats: rural, small town, suburban, urban stores sell different amounts
  * day of week: weekends are busier
  * first-of-month / SNAP window: food sales jump on days 1-10 when benefits arrive
  * paycheck Fridays: every other Friday is busier
  * holidays: the week before is busier, the day itself is quieter
  * seasons: beverages peak in summer
  * prices and promotions: higher prices lower demand (elasticity); promos lift it
  * weather: storms cut demand in a region for a few days
  * new stores: some stores open part-way through, so they have short histories
  * stockouts: when a shelf is empty, recorded sales understate true demand

Outputs (in artifacts/data/raw):
  stores.parquet, items.parquet, calendar.parquet
  sales_history.parquet   - two years of daily sales the model trains on
  future_plan.parquet     - next 4 weeks of prices/promos (known in advance, no sales)
  future_actuals.parquet  - what really sold in those 4 weeks (arrives later; used to monitor)
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from dgforecast.config import get_paths, load_config

FORMATS = ["rural", "small_town", "suburban", "urban"]
FORMAT_P = [0.40, 0.30, 0.20, 0.10]
FORMAT_SCALE = {"rural": 0.7, "small_town": 1.0, "suburban": 1.3, "urban": 1.6}
REGIONS = ["South", "Southeast", "Midwest", "Southwest"]
CATEGORIES = ["snacks", "beverages", "household", "paper", "health_beauty", "frozen", "dairy", "seasonal"]
FOOD = {"snacks", "beverages", "frozen", "dairy"}
PRICE_RANGE = {"snacks": (1.25, 3), "beverages": (1, 2.5), "household": (3, 8), "paper": (4, 12),
               "health_beauty": (3, 10), "frozen": (2.5, 6), "dairy": (2, 5), "seasonal": (3, 15)}
ELASTICITY = {"snacks": -1.4, "beverages": -1.5, "household": -0.8, "paper": -0.9,
              "health_beauty": -1.0, "frozen": -1.2, "dairy": -0.7, "seasonal": -1.6}
SUMMER_AMP = {"beverages": 0.25, "frozen": 0.15}
DOW = np.array([0.90, 0.85, 0.90, 0.95, 1.15, 1.25, 1.05])  # Mon..Sun


def _holidays(years) -> list[date]:
    out = []
    for y in years:
        out += [date(y, 1, 1), date(y, 7, 4), date(y, 12, 25)]
        # Memorial Day: last Monday of May
        d = date(y, 5, 31)
        out.append(d - timedelta(days=d.weekday()))
        # Labor Day: first Monday of September
        d = date(y, 9, 1)
        out.append(d + timedelta(days=(7 - d.weekday()) % 7))
        # Thanksgiving: fourth Thursday of November
        d = date(y, 11, 1)
        first_thu = d + timedelta(days=(3 - d.weekday()) % 7)
        out.append(first_thu + timedelta(weeks=3))
    return out


def build_calendar(start: str, n: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="D")
    cal = pd.DataFrame({"date": dates})
    cal["dow"] = cal.date.dt.dayofweek
    cal["dom"] = cal.date.dt.day
    cal["month"] = cal.date.dt.month
    cal["is_snap_window"] = (cal.dom <= 10).astype(int)
    fridays = cal.index[cal.dow == 4]
    cal["is_paycheck_friday"] = 0
    cal.loc[fridays[::2], "is_paycheck_friday"] = 1
    hol = pd.to_datetime(_holidays(range(dates[0].year, dates[-1].year + 2)))
    cal["is_holiday"] = cal.date.isin(hol).astype(int)
    # days until the next holiday (capped at 30): the week before a holiday is busy
    nxt = np.searchsorted(hol.values, cal.date.values)
    days_to = (hol.values[np.minimum(nxt, len(hol) - 1)] - cal.date.values) / np.timedelta64(1, "D")
    cal["days_to_holiday"] = np.clip(days_to, 0, 30).astype(int)
    return cal


def generate(cfg: dict) -> dict[str, pd.DataFrame]:
    c = cfg["data"]
    rng = np.random.default_rng(c["seed"])
    S, I = c["n_stores"], c["n_items"]
    D = c["n_days"] + c["future_days"]
    cal = build_calendar(c["start_date"], D)

    # ---- stores ----
    stores = pd.DataFrame({
        "store_id": [f"S{n:03d}" for n in range(S)],
        "format": rng.choice(FORMATS, S, p=FORMAT_P),
        "region": rng.choice(REGIONS, S, p=[0.4, 0.25, 0.2, 0.15]),
    })
    open_day = np.where(rng.random(S) < 0.85, 0, rng.integers(60, c["n_days"] - 90, S))
    stores["open_date"] = cal.date.values[open_day]

    # ---- items ----
    cats = rng.choice(CATEGORIES, I)
    lo = np.array([PRICE_RANGE[k][0] for k in cats]); hi = np.array([PRICE_RANGE[k][1] for k in cats])
    items = pd.DataFrame({
        "item_id": [f"I{n:04d}" for n in range(I)],
        "category": cats,
        "base_price": np.round(rng.uniform(lo, hi), 2),
    })
    popularity = 3.0 * rng.lognormal(0, 0.6, I)

    # ---- prices and promotions (chain-wide, per item per day) ----
    price = np.repeat(items.base_price.values[:, None], D, axis=1).astype(float)
    for i in range(I):  # 0-2 permanent price increases per item
        for _ in range(rng.integers(0, 3)):
            price[i, rng.integers(30, D):] *= 1 + rng.uniform(0.05, 0.10)
    n_weeks = D // 7 + 1
    promo_week = rng.random((I, n_weeks)) < 0.06
    on_promo = np.repeat(promo_week, 7, axis=1)[:, :D].astype(int)
    shelf_price = price * np.where(on_promo == 1, 0.8, 1.0)

    # drift scenario: household price increase in the future window
    dr = c.get("drift_scenario", {}) or {}
    future0 = c["n_days"]
    if dr.get("enabled"):
        for cat, chg in (dr.get("category_price_change") or {}).items():
            shelf_price[cats == cat, future0:] *= 1 + chg
            price[cats == cat, future0:] *= 1 + chg

    elas = np.array([ELASTICITY[k] for k in cats])
    price_effect = (shelf_price / items.base_price.values[:, None]) ** elas[:, None]
    promo_lift = np.where(on_promo == 1, 1.15, 1.0)

    # ---- calendar effects (item x day) ----
    dow = DOW[cal.dow.values][None, :]
    is_food = np.isin(cats, list(FOOD))
    snap = 1 + np.where(is_food[:, None], 0.18, 0.06) * cal.is_snap_window.values[None, :]
    payday = 1 + 0.08 * cal.is_paycheck_friday.values[None, :]
    pre_hol = ((cal.days_to_holiday.values >= 1) & (cal.days_to_holiday.values <= 6))[None, :]
    hol = 1 + np.where(cats[:, None] == "seasonal", 0.40, 0.15) * pre_hol
    hol = hol * np.where(cal.is_holiday.values[None, :] == 1, 0.8, 1.0)
    doy = cal.date.dt.dayofyear.values
    summer = np.array([SUMMER_AMP.get(k, 0.05) for k in cats])[:, None] * np.sin(2 * np.pi * (doy - 100) / 365)[None, :]
    trend = (1 + 0.05 * np.arange(D) / 365)[None, :]
    item_day = popularity[:, None] * dow * snap * payday * hol * (1 + summer) * trend * price_effect * promo_lift

    # ---- store effects (store x day) ----
    store_scale = np.array([FORMAT_SCALE[f] for f in stores.format]) * rng.lognormal(0, 0.15, S)
    weather = np.ones((len(REGIONS), D))
    for r in range(len(REGIONS)):
        for d0 in rng.integers(0, D - 3, int(6 * D / 365)):
            weather[r, d0:d0 + rng.integers(1, 4)] = 0.65
    reg_idx = np.array([REGIONS.index(r) for r in stores.region])
    store_day = store_scale[:, None] * weather[reg_idx]
    if dr.get("enabled") and dr.get("region_hit"):
        store_day[stores.region.values == dr["region_hit"], future0:] *= 1 + dr["region_demand_change"]

    # ---- true demand: rate -> gamma-Poisson (overdispersed counts) ----
    rate = store_day[:, None, :] * item_day[None, :, :]  # S x I x D
    lam = rng.gamma(shape=5.0, scale=rate / 5.0)
    demand = rng.poisson(lam)

    # ---- stockouts: empty shelf for 1-4 days; recorded sales understate demand ----
    start = rng.random((S, I, D)) < 0.006
    length = rng.integers(1, 5, (S, I, D))
    flag = np.zeros((S, I, D), dtype=bool)
    for k in range(4):
        flag[..., k:] |= (start & (length > k))[..., : D - k]
    sold = np.where(flag, np.floor(demand * rng.uniform(0, 0.35, (S, I, D))), demand).astype(np.int32)

    # ---- to a long table, dropping days before a store opened ----
    s_idx, i_idx, d_idx = np.meshgrid(np.arange(S), np.arange(I), np.arange(D), indexing="ij")
    keep = (d_idx >= open_day[:, None, None]).ravel()
    sales = pd.DataFrame({
        "date": cal.date.values[d_idx.ravel()[keep]],
        "store_id": stores.store_id.values[s_idx.ravel()[keep]],
        "item_id": items.item_id.values[i_idx.ravel()[keep]],
        "units_sold": sold.ravel()[keep],
        "price": np.round(shelf_price[i_idx.ravel()[keep], d_idx.ravel()[keep]], 2),
        "on_promo": on_promo[i_idx.ravel()[keep], d_idx.ravel()[keep]],
        "stockout_flag": flag.ravel()[keep].astype(int),
    })
    # The simulator knows each day's true expected demand. A real company never has this;
    # we keep it apart (never a model input) to measure how good a *perfect* model could be.
    truth = sales[["date", "store_id", "item_id"]].assign(expected_demand=rate.ravel()[keep])
    cutoff = cal.date.values[future0]
    history = sales[sales.date < cutoff].reset_index(drop=True)
    future = sales[sales.date >= cutoff].reset_index(drop=True)
    plan = future.drop(columns=["units_sold", "stockout_flag"])
    return {"stores": stores, "items": items, "calendar": cal,
            "sales_history": history, "future_plan": plan, "future_actuals": future,
            "simulator_truth": truth}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate synthetic dollar-store sales.")
    ap.add_argument("--config"); ap.add_argument("--out")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    paths = get_paths(cfg, a.out)
    tables = generate(cfg)
    for name, df in tables.items():
        df.to_parquet(paths.raw / f"{name}.parquet", index=False)
    h = tables["sales_history"]
    print(f"history: {len(h):,} rows, {h.store_id.nunique()} stores, {h.item_id.nunique()} items, "
          f"{h.date.min().date()} to {h.date.max().date()}; stockout days {h.stockout_flag.mean():.1%}")


if __name__ == "__main__":
    main()
