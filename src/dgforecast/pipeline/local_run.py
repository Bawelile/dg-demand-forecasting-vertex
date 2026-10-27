"""Run the whole training pipeline on your laptop, step by step.

    generate -> validate -> features -> train -> evaluate + gate -> register -> batch forecast -> explain

Each step is a separate module you can also run alone (see the Makefile). The Vertex AI
pipeline (vertex_pipeline.py) runs these same steps as containers on Google Cloud.
"""
from __future__ import annotations

import argparse
import time

from dgforecast.data import generate, validate
from dgforecast.evaluate import evaluate
from dgforecast.explain import explain
from dgforecast.features import build
from dgforecast.predict import batch
from dgforecast.train import train


def step(name, fn, args):
    t = time.time(); print(f"\n=== {name} ===")
    fn(args)
    print(f"--- {name} done in {time.time() - t:.1f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--out")
    ap.add_argument("--skip-generate", action="store_true")
    a = ap.parse_args(argv)
    common = (["--config", a.config] if a.config else []) + (["--out", a.out] if a.out else [])
    if not a.skip_generate:
        step("1. generate synthetic data", generate.main, common)
    step("2. validate data", validate.main, common)
    step("3. build features", build.main, common)
    step("4. train XGBoost", train.main, common)
    step("5. evaluate, gate, register", evaluate.main, common)
    step("6. batch forecast", batch.main, common)
    step("7. explain", explain.main, common)


if __name__ == "__main__":
    main()
