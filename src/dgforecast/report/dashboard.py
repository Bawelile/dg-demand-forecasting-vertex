"""Build dashboard.html: one page that shows every part of the system working together.

Reads the real outputs in artifacts/ (validation, evaluation, forecasts, monitoring,
registry, experiments) plus short excerpts of the code, and writes them as JSON into
report/template.html. Rerun it after any pipeline run and the page updates.

    python -m dgforecast.report.dashboard      ->  dashboard.html (repo root, for GitHub Pages)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from dgforecast.config import REPO_ROOT, get_paths, load_config, read_json
from dgforecast.predict.batch import READABLE

SRC = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("template.html")


def excerpt(rel: str, start: str, n: int) -> str:
    """n lines of a source file, starting at the first line containing `start`."""
    lines = (SRC / rel).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if start in l)
    block = lines[i:i + n]
    pad = min(len(l) - len(l.lstrip()) for l in block if l.strip())
    return "\n".join(l[pad:] for l in block)


SNIPPETS = [
    ("data", "data/generate.py", "    snap = 1 + np.where", 6),
    ("validate", "data/validate.py", "    dups = int(", 6),
    ("features", "features/build.py", "    clean = df[\"units_sold\"].where", 8),
    ("train", "train/train.py", "    tr, va, te = time_split(", 7),
    ("gate", "evaluate/evaluate.py", "        needed = ch * (1 -", 2),
    ("forecast", "predict/batch.py", "    feats[\"predicted_units\"] = np.clip(", 3),
    ("explain", "predict/batch.py", "    contrib = booster.predict(", 2),
    ("monitor", "monitor/monitor.py", "    blim = cfg[\"monitoring\"][\"region_bias_alert\"]", 2),
    ("pipeline", "pipeline/vertex_pipeline.py", "    v = validate_op(", 11),
]


def collect(cfg: dict, paths) -> dict:
    raw = paths.raw
    stores = pd.read_parquet(raw / "stores.parquet")
    items = pd.read_parquet(raw / "items.parquet")
    hist_f = raw / "sales_history_before_retrain.parquet"
    hist = pd.read_parquet(hist_f if hist_f.exists() else raw / "sales_history.parquet")
    future = pd.read_parquet(raw / "future_actuals.parquet")

    evals = sorted(paths.reports.glob("evaluation_run-*.json"))
    ev1 = read_json(evals[0])
    run1 = read_json(paths.models / ev1["run_id"] / "run_info.json")
    fmt_names = {"rural": "Rural", "small_town": "Small town", "suburban": "Suburban", "urban": "Urban"}

    def seg(d, names=None):
        return [{"name": (names or {}).get(k, k.replace("_", " ").capitalize()), "wape": v["wape"], "bias": v["bias"]}
                for k, v in d.items()]

    c, b = ev1["candidate"], ev1["baseline"]
    data = {
        "meta": {"stores": int(stores.shape[0]), "items": int(items.shape[0]), "rows": int(len(hist)),
                 "start": str(hist.date.min().date()), "end": str(hist.date.max().date()),
                 "horizon": cfg["forecast"]["horizon_days"], "features": len(run1["features"]),
                 "train_rows": run1["train_rows"], "trees": run1["best_iteration"],
                 "train_dates": run1["train_dates"], "valid_dates": run1["valid_dates"], "test_dates": run1["test_dates"],
                 "stockout_rate": float(hist.stockout_flag.mean())},
        "validation": read_json(paths.reports / "validation.json")["checks"],
        "eval": {
            "daily": {"baseline": b["overall"]["wape"], "model": c["overall"]["wape"], "oracle": ev1.get("oracle_wape")},
            "weekly": {"baseline": ev1["weekly_wape"]["baseline"], "model": ev1["weekly_wape"]["candidate"]},
            "bias": {"baseline": b["overall"]["bias"], "model": c["overall"]["bias"]},
            "dollars": {k: {"lost": v["dollars"]["lost_margin_usd"], "holding": v["dollars"]["holding_cost_usd"]}
                        for k, v in (("baseline", b), ("model", c))},
            "by_format": seg(c["by_store_format"], fmt_names),
            "by_category": seg(c["by_category"]),
            "decision": ev1["decision"],
        },
        "importance": [{"name": k, "value": v} for k, v in list(read_json(paths.reports / "feature_importance.json").items())[:12]],
    }

    # forecast explorer: store totals (one per region) and two single items
    fc = pd.read_parquet(max(paths.predictions.glob("forecast_*.parquet")))
    origin = pd.Timestamp(fc.forecast_origin.iloc[0])
    h56 = hist[hist.date > origin - pd.Timedelta(days=56)]
    act14 = future[future.date <= origin + pd.Timedelta(days=cfg["forecast"]["horizon_days"])]
    series = []

    def add(label, sub, mask_h, mask_f, mask_a, reasons=False):
        hh = h56[mask_h(h56)].groupby("date").units_sold.sum()
        ff = fc[mask_f(fc)].groupby("target_date").predicted_units.sum()
        aa = act14[mask_a(act14)].groupby("date").units_sold.sum()
        pts = []
        for d, yhat in ff.items():
            p = {"d": str(d.date()), "yhat": round(float(yhat), 1), "actual": int(aa.get(d, 0))}
            if reasons:
                r = fc[mask_f(fc) & (fc.target_date == d)].iloc[0]
                p["reasons"] = [r.driver_1, r.driver_2, r.driver_3]
            pts.append(p)
        series.append({"label": label, "sub": sub,
                       "history": [{"d": str(d.date()), "y": int(v)} for d, v in hh.items()], "forecast": pts})

    st = stores.set_index("store_id")
    for region in ["Midwest", "Southeast", "Southwest", "South"]:
        sid = st[st.region == region].index[0]
        add(f"Store {sid}, all items", f"{region} · {fmt_names[st.loc[sid, 'format']]}",
            lambda d, s=sid: d.store_id == s, lambda d, s=sid: d.store_id == s, lambda d, s=sid: d.store_id == s)
    top = fc.groupby(["store_id", "item_id"]).predicted_units.sum().sort_values(ascending=False)
    cat_of = items.set_index("item_id").category
    not_south = [k for k in top.index if st.loc[k[0], "region"] != "South"][0]
    south_household = [k for k in top.index if st.loc[k[0], "region"] == "South" and cat_of[k[1]] == "household"][0]
    for (sid, iid) in (not_south, south_household):
        cat = items.set_index("item_id").loc[iid, "category"].replace("_", " ")
        add(f"Store {sid}, item {iid}", f"{st.loc[sid, 'region']} · {cat} · with reasons",
            lambda d, s=sid, i=iid: (d.store_id == s) & (d.item_id == i),
            lambda d, s=sid, i=iid: (d.store_id == s) & (d.item_id == i),
            lambda d, s=sid, i=iid: (d.store_id == s) & (d.item_id == i), reasons=True)
    data["series"] = series
    data["origin"] = str(origin.date())

    mon_f = max(paths.reports.glob("monitoring_*.json"))
    mon = read_json(mon_f)
    od = mon["outcome_decay"]
    data["monitoring"] = {
        "psi": [{"name": k.replace("_", " ").capitalize(), "psi": v} for k, v in mon["input_drift"]["price_index_by_category"].items()],
        "psi_limit": cfg["monitoring"]["psi_alert"],
        "regions": [{"name": k, "wape": v["wape"], "bias": v["bias"], "alert": k in od.get("regions_off", [])}
                    for k, v in od["by_region"].items()],
        "bias_limit": cfg["monitoring"]["region_bias_alert"],
        "live_wape": od["live_wape"], "training_wape": od["training_wape"], "threshold": od["alert_threshold"],
        "flagged": mon["input_drift"]["flagged"], "scenario": cfg["data"].get("drift_scenario", {}),
    }
    if len(evals) > 1:
        ev2 = read_json(evals[-1])
        data["retrain"] = {"candidate": ev2["candidate"]["overall"]["wape"], "champion": ev2["champion"]["overall"]["wape"],
                           "baseline": ev2["baseline"]["overall"]["wape"], "promote": ev2["decision"]["promote"],
                           "reason": ev2["decision"]["reason"]}
    data["registry"] = []
    ch = read_json(paths.registry / "champion.json")
    for v in sorted(paths.registry.glob("v*")):
        card = read_json(v / "model_card.json")
        data["registry"].append({"version": card["version"], "run_id": card["run_id"], "trained": card["trained_at"][:10],
                                 "wape": card["metrics"]["wape"], "champion": card["version"] == ch["version"],
                                 "train": card["data"]["train"], "test": card["data"]["test"]})
    with open(paths.reports / "experiments.csv") as f:
        data["experiments"] = list(csv.DictReader(f))
    data["snippets"] = {k: {"file": "src/dgforecast/" + f, "code": excerpt(f, s, n)} for k, f, s, n in SNIPPETS}
    return data


def build(cfg: dict, paths, out: Path) -> Path:
    data = collect(cfg, paths)
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, default=str))
    out.write_text(html)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--out")
    ap.add_argument("--html", default=str(REPO_ROOT / "dashboard.html"))
    a = ap.parse_args(argv)
    cfg = load_config(a.config); paths = get_paths(cfg, a.out)
    print("wrote", build(cfg, paths, Path(a.html)))


if __name__ == "__main__":
    main()
