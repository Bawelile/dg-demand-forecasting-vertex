"""The most important test in the project: features must not see the future."""
import pandas as pd

from dgforecast.data.generate import generate
from dgforecast.features.build import FEATURES, build_features


def test_no_leakage_within_forecast_horizon(small_cfg):
    t = generate(small_cfg)
    sales, stores, items, cal = t["sales_history"], t["stores"], t["items"], t["calendar"]
    target_day = sales.date.max() - pd.Timedelta(days=30)
    before = build_features(sales, stores, items, cal, small_cfg)

    # Change every sale in the 13 days before target_day, and on target_day itself.
    # Those days are unknown when we forecast 14 days ahead, so features must not move.
    changed = sales.copy()
    window = (changed.date > target_day - pd.Timedelta(days=14)) & (changed.date <= target_day)
    changed.loc[window, "units_sold"] += 50
    after = build_features(changed, stores, items, cal, small_cfg)

    a = before[before.date == target_day].set_index(["store_id", "item_id"])[FEATURES]
    b = after[after.date == target_day].set_index(["store_id", "item_id"])[FEATURES]
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_stockout_days_are_blanked(small_cfg):
    t = generate(small_cfg)
    feats = build_features(t["sales_history"], t["stores"], t["items"], t["calendar"], small_cfg)
    assert feats["lag_14"].isna().any()  # stockout days 14 days earlier become blanks, not zeros
