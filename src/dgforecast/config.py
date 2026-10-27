"""Load config.yaml and give every module the same folder layout."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


@dataclass
class Paths:
    """Where each step reads and writes. Locally these are folders;
    on Google Cloud the same names become Cloud Storage and BigQuery locations."""

    root: Path

    @property
    def raw(self) -> Path: return self.root / "data" / "raw"
    @property
    def features(self) -> Path: return self.root / "data" / "features"
    @property
    def models(self) -> Path: return self.root / "models"
    @property
    def registry(self) -> Path: return self.root / "registry"
    @property
    def predictions(self) -> Path: return self.root / "predictions"
    @property
    def reports(self) -> Path: return self.root / "reports"

    def make(self) -> "Paths":
        for p in (self.raw, self.features, self.models, self.registry, self.predictions, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(cfg: dict, root: str | Path | None = None) -> Paths:
    base = Path(root) if root else REPO_ROOT / cfg["paths"]["root"]
    return Paths(base).make()


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)
