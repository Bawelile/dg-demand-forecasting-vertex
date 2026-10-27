import pandas as pd

from dgforecast.data.generate import generate
from dgforecast.data.validate import validate


def test_clean_data_passes(small_cfg):
    sales = generate(small_cfg)["sales_history"]
    assert validate(sales, small_cfg)["passed"]


def test_duplicates_and_negative_sales_fail(small_cfg):
    sales = generate(small_cfg)["sales_history"]
    bad = sales.copy()
    bad = pd.concat([bad, bad.iloc[[0]]], ignore_index=True)   # a duplicated row
    bad.loc[5, "units_sold"] = -3                           # an impossible value
    report = validate(bad, small_cfg)
    failed = {c["check"] for c in report["checks"] if not c["passed"]}
    assert not report["passed"]
    assert {"duplicates", "non_negative_units"} <= failed
