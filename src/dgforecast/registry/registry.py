"""Lesson 6 (part 2). A model registry: every approved model gets a version number,
and one version is marked "champion" (the one in production).

Locally this is a folder:
    artifacts/registry/v1/, v2/, ...  (model.json + model_card.json)
    artifacts/registry/champion.json  (which version is live)
On Google Cloud the same idea is Vertex AI Model Registry: upload a new version under one
model name and give the live one the alias "champion" (see cloud/vertex_registry.py).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import xgboost as xgb

from dgforecast.config import read_json, write_json


def champion(paths) -> dict | None:
    f = paths.registry / "champion.json"
    return read_json(f) if f.exists() else None


def load_champion_model(paths) -> tuple[xgb.Booster, dict] | tuple[None, None]:
    ch = champion(paths)
    if not ch:
        return None, None
    b = xgb.Booster(); b.load_model(paths.registry / ch["version"] / "model.json")
    return b, ch


def register(run_dir: Path, evaluation: dict, paths, promote: bool) -> dict:
    versions = sorted(int(p.name[1:]) for p in paths.registry.glob("v*") if p.is_dir())
    version = f"v{(versions[-1] + 1) if versions else 1}"
    dest = paths.registry / version
    dest.mkdir(parents=True)
    shutil.copy(run_dir / "model.json", dest / "model.json")
    info = read_json(run_dir / "run_info.json")
    card = {
        "version": version, "run_id": info["run_id"], "trained_at": info["trained_at"],
        "intended_use": "14-day store-item unit forecasts for replenishment planning",
        "data": {"train": info["train_dates"], "test": info["test_dates"], "rows": info["train_rows"]},
        "metrics": evaluation["candidate"]["overall"],
        "baseline_metrics": evaluation["baseline"]["overall"],
        "limitations": [
            "Trained on synthetic data; validate on real sales before use.",
            "New stores rely on chain-level item averages until they have 4+ weeks of history.",
            "Stockout days are excluded from training, so forecasts estimate demand, not recorded sales.",
        ],
        "status": "champion" if promote else "registered (not promoted)",
    }
    write_json(card, dest / "model_card.json")
    if promote:
        write_json({"version": version, "run_id": info["run_id"],
                    "test_wape": evaluation["candidate"]["overall"]["wape"]}, paths.registry / "champion.json")
    return card
