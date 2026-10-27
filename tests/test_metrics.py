from dgforecast.evaluate.metrics import bias, dollar_impact, wape


def test_wape_and_bias():
    y, yhat = [10, 0, 5], [8, 1, 5]
    assert wape(y, yhat) == (2 + 1 + 0) / 15
    assert bias(y, yhat) == (14 - 15) / 15


def test_perfect_forecast_scores_zero():
    assert wape([3, 4], [3, 4]) == 0 and bias([3, 4], [3, 4]) == 0


def test_dollar_impact_splits_under_and_over():
    d = dollar_impact(y=[10, 10], yhat=[8, 13], price=[2, 2], margin_rate=0.5, holding_rate=0.25)
    assert d["lost_margin_usd"] == 2 * 2 * 0.5      # 2 units short
    assert d["holding_cost_usd"] == 3 * 2 * 0.25    # 3 units extra
