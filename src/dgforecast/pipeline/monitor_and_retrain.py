"""Lessons 9-10. Simulate four weeks passing in production, then monitor and retrain.

1. Actual sales for the forecast period arrive (they include the drift scenario from
   config: a new competitor in the South and a 12% household price increase).
2. Monitoring compares inputs (PSI) and outcomes (WAPE, bias) with training time.
3. If any alert fires, the new weeks are appended to history and the full pipeline reruns:
   validate -> features -> train -> evaluate + gate. The gate still decides: a retrained
   model only replaces the champion if it is measurably better on the newest weeks.
"""
from __future__ import annotations

import argparse
import shutil

import pandas as pd

from dgforecast.config import get_paths, load_config, write_json
from dgforecast.data import validate
from dgforecast.data.generate import CATEGORIES
from dgforecast.evaluate import evaluate
from dgforecast.features import build
from dgforecast.monitor.monitor import input_drift, outcome_decay
from dgforecast.registry.registry import champion
from dgforecast.train import train


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--out")
    a = ap.parse_args(argv)
    common = (["--config", a.config] if a.config else []) + (["--out", a.out] if a.out else [])
    cfg = load_config(a.config); paths = get_paths(cfg, a.out)

    fc_file = max(paths.predictions.glob("forecast_*.parquet"))
    tag = fc_file.stem.split("_")[1]
    ch = champion(paths)
    print(f"=== Monitoring forecast {tag} from champion {ch['version']} ===")
    ref = pd.read_parquet(paths.models / ch["run_id"] / "reference_sample.parquet")
    scoring = pd.read_parquet(paths.predictions / f"scoring_features_{tag}.parquet")
    drift = input_drift(ref, scoring, cfg, dict(enumerate(CATEGORIES)))
    actuals = pd.read_parquet(paths.raw / "future_actuals.parquet")
    stores = pd.read_parquet(paths.raw / "stores.parquet")
    decay = outcome_decay(pd.read_parquet(fc_file), actuals, stores, cfg, paths)
    retrain = bool(drift["flagged"] or decay["alert"])
    report = {"forecast": tag, "champion": ch["version"], "input_drift": drift,
              "outcome_decay": decay, "retrain_triggered": retrain}
    write_json(report, paths.reports / f"monitoring_{tag}.json")
    print("input drift flagged:", drift["flagged"] or "none")
    print(f"live WAPE {decay['live_wape']:.1%} vs training {decay['training_wape']:.1%} "
          f"(alert above {decay['alert_threshold']:.1%}); bias {decay['live_bias']:+.1%}")
    for r, m in decay["by_region"].items():
        flag = "  <-- ALERT" if r in decay["regions_off"] else ""
        print(f"   {r:10s} WAPE {m['wape']:.1%}  bias {m['bias']:+.1%}{flag}")
    if not retrain:
        print("No alert. Champion stays; next scheduled retrain is next week.")
        return report

    print("\n=== Alert fired: retraining with the new weeks of actuals ===")
    hist_f = paths.raw / "sales_history.parquet"
    backup = paths.raw / "sales_history_before_retrain.parquet"
    if not backup.exists():
        shutil.copy(hist_f, backup)
    hist = pd.read_parquet(backup)
    pd.concat([hist, actuals], ignore_index=True).to_parquet(hist_f, index=False)
    validate.main(common)
    build.main(common)
    train.main(common)
    evaluate.main(common)
    report["new_champion"] = champion(paths)["version"]
    write_json(report, paths.reports / f"monitoring_{tag}.json")
    return report


if __name__ == "__main__":
    main()
