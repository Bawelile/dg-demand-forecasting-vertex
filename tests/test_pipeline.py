"""End to end on tiny data: train, gate, register, forecast, and compile the Vertex pipeline."""
import yaml

from dgforecast.config import get_paths, read_json
from dgforecast.pipeline import local_run


def test_local_pipeline_runs_and_forecasts(small_cfg, tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(small_cfg))
    out = tmp_path / "artifacts"
    local_run.main(["--config", str(cfg_file), "--out", str(out)])

    paths = get_paths(small_cfg, out)
    champ = read_json(paths.registry / "champion.json")
    assert champ["version"] == "v1"
    ev = read_json(next(paths.reports.glob("evaluation_*.json")))
    assert ev["candidate"]["overall"]["wape"] < ev["baseline"]["overall"]["wape"]

    import pandas as pd
    fc = pd.read_parquet(next(paths.predictions.glob("forecast_*.parquet")))
    n = small_cfg["data"]["n_stores"] * small_cfg["data"]["n_items"] * small_cfg["forecast"]["horizon_days"]
    assert len(fc) == n
    assert (fc.predicted_units >= 0).all()
    assert fc.driver_1.notna().all()


def test_vertex_pipeline_compiles(tmp_path):
    from dgforecast.pipeline.vertex_pipeline import compile_all
    files = compile_all(str(tmp_path))
    assert all((tmp_path / f.split("/")[-1]).exists() for f in files)
