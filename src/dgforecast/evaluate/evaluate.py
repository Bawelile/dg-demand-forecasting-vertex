"""Lesson 6. Evaluate the new model and decide if it may replace the live one (the gate).

Three models are scored on exactly the same 28 test days:
  1. baseline: "same weekday two weeks ago" (seasonal naive). Any real model must beat it.
  2. champion: the model in production today (if there is one).
  3. candidate: the model we just trained.

Gate rule (from config): promote the candidate only if it beats the baseline AND improves
WAPE over the champion by at least 1%. Otherwise keep the champion. This is how you stop
a worse model from ever reaching stores.

We also score by store format and category, because an overall average can hide a model
that is great in suburbs and bad in rural stores.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xgboost as xgb

from dgforecast.config import get_paths, load_config, read_json, write_json
from dgforecast.data.generate import CATEGORIES, FORMATS
from dgforecast.evaluate.metrics import dollar_impact, scorecard, wape
from dgforecast.features.build import FEATURES
from dgforecast.registry.registry import load_champion_model, register


def seasonal_naive(test: pd.DataFrame) -> pd.Series:
    """Same weekday two weeks ago; fall back to the 28-day average if that day was a stockout."""
    return test["lag_14"].fillna(test["rmean_28"]).fillna(0)


def evaluate(run_dir: Path, cfg: dict, paths) -> dict:
    test = pd.read_parquet(run_dir / "test_predictions.parquet")
    test["baseline"] = seasonal_naive(test)
    champ, ch_info = load_champion_model(paths)
    if champ is not None:
        test["champion"] = champ.predict(xgb.DMatrix(test[FEATURES], feature_names=FEATURES))
    e = cfg["economics"]
    out = {"run_id": run_dir.name}
    for name, col in [("candidate", "pred"), ("baseline", "baseline")] + ([("champion", "champion")] if champ else []):
        seg = test.assign(store_format=test.store_format.map(dict(enumerate(FORMATS))),
                          category=test.category.map(dict(enumerate(CATEGORIES))))
        out[name] = {
            "overall": scorecard(test, col),
            "by_store_format": scorecard(seg, col, "store_format").round(4).to_dict("index"),
            "by_category": scorecard(seg, col, "category").round(4).to_dict("index"),
            "dollars": dollar_impact(test.units_sold, test[col], test.price,
                                     e["gross_margin_rate"], e["holding_cost_rate"]),
        }
    # Weekly WAPE: replenishment orders by the week, and daily noise partly cancels out.
    wk = test.assign(week=test.date.dt.to_period("W").dt.start_time)
    agg = wk.groupby(["store_id", "item_id", "week"])[["units_sold", "pred", "baseline"]].sum()
    out["weekly_wape"] = {"candidate": wape(agg.units_sold, agg.pred),
                          "baseline": wape(agg.units_sold, agg.baseline)}
    # Noise floor: how a perfect model (the simulator's true expected demand) would score.
    truth_f = paths.raw / "simulator_truth.parquet"
    if truth_f.exists():
        tr = pd.read_parquet(truth_f)
        t = test[["date", "store_id", "item_id", "units_sold"]].merge(tr, on=["date", "store_id", "item_id"])
        out["oracle_wape"] = wape(t.units_sold, t.expected_demand)
    cand = out["candidate"]["overall"]["wape"]
    base = out["baseline"]["overall"]["wape"]
    beats_baseline = cand < base
    if champ is None:
        promote = beats_baseline
        reason = "no champion yet; candidate beats baseline" if promote else "candidate does not beat baseline"
    else:
        ch = out["champion"]["overall"]["wape"]
        needed = ch * (1 - cfg["gate"]["min_relative_improvement"])
        promote = beats_baseline and cand <= needed
        reason = (f"candidate WAPE {cand:.2%} vs champion {ch:.2%} (needs <= {needed:.2%}); "
                  f"baseline {base:.2%}")
    out["decision"] = {"promote": bool(promote), "reason": reason,
                       "champion_version": ch_info["version"] if ch_info else None}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--out")
    ap.add_argument("--run-dir", help="defaults to the newest run in artifacts/models")
    a = ap.parse_args(argv)
    cfg = load_config(a.config); paths = get_paths(cfg, a.out)
    run_dir = Path(a.run_dir) if a.run_dir else max(paths.models.glob("run-*"), key=lambda p: p.name)
    ev = evaluate(run_dir, cfg, paths)
    write_json(ev, paths.reports / f"evaluation_{run_dir.name}.json")
    card = register(run_dir, ev, paths, promote=ev["decision"]["promote"])
    floor = f" | perfect-model floor {ev['oracle_wape']:.1%}" if "oracle_wape" in ev else ""
    print(f"daily WAPE: baseline {ev['baseline']['overall']['wape']:.1%} | "
          f"candidate {ev['candidate']['overall']['wape']:.1%}{floor}")
    print(f"weekly WAPE: baseline {ev['weekly_wape']['baseline']:.1%} | candidate {ev['weekly_wape']['candidate']:.1%}")
    d_c, d_b = ev["candidate"]["dollars"]["total_cost_usd"], ev["baseline"]["dollars"]["total_cost_usd"]
    print(f"cost of forecast error over the test weeks: model ${d_c:,.0f} vs baseline ${d_b:,.0f}")
    print(f"gate: "
          f"{'PROMOTED' if ev['decision']['promote'] else 'NOT promoted'} as {card['version']}: "
          f"{ev['decision']['reason']}")


if __name__ == "__main__":
    main()
