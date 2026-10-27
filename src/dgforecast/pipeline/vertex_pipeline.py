"""Lesson 10. The same steps as local_run.py, as a Vertex AI Pipeline (Kubeflow Pipelines v2).

Each step runs in a container built from this repo's Dockerfile. Steps read and write the
artifacts folder on Cloud Storage through Vertex AI's /gcs/ mount, so the code is unchanged
from your laptop: only --out points to /gcs/<bucket>/... instead of ./artifacts.

Training pipeline (weekly schedule):
    validate -> features -> train -> evaluate + gate --(promote?)--> upload to Model Registry
                                                                  -> batch forecast -> BigQuery
Monitoring pipeline (after actuals arrive):
    monitor -> (alert?) -> logs an ERROR that a Cloud Monitoring alert policy watches,
               and starts the training pipeline again

Compile (no Google Cloud account needed):  python -m dgforecast.pipeline.vertex_pipeline
Submit and schedule:                        python -m dgforecast.cloud.submit
"""

import os

from kfp import compiler, dsl

from dgforecast.config import load_config

CFG = load_config()
IMAGE = os.environ.get("DGFORECAST_IMAGE", CFG["project"]["image"])


def _cli(module: str, out_root: str):
    return dsl.ContainerSpec(image=IMAGE, command=["python", "-m", module], args=["--out", out_root])


@dsl.container_component
def validate_op(out_root: str):
    return _cli("dgforecast.data.validate", out_root)


@dsl.container_component
def features_op(out_root: str):
    return _cli("dgforecast.features.build", out_root)


@dsl.container_component
def train_op(out_root: str):
    return _cli("dgforecast.train.train", out_root)


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def evaluate_op(out_root: str) -> str:
    """Score candidate vs champion vs baseline, register, and return 'true' if promoted."""
    from dgforecast.config import get_paths, load_config, write_json
    from dgforecast.evaluate.evaluate import evaluate
    from dgforecast.registry.registry import register
    cfg = load_config(); paths = get_paths(cfg, out_root)
    run_dir = max(paths.models.glob("run-*"), key=lambda p: p.name)
    ev = evaluate(run_dir, cfg, paths)
    write_json(ev, paths.reports / f"evaluation_{run_dir.name}.json")
    register(run_dir, ev, paths, promote=ev["decision"]["promote"])
    print(ev["decision"])
    return "true" if ev["decision"]["promote"] else "false"


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def upload_model_op(out_root: str, project: str, region: str, display_name: str, serving_image: str) -> str:
    """Upload the new champion to Vertex AI Model Registry as a new version with alias 'champion'."""
    from dgforecast.cloud.vertex_registry import upload_champion
    return upload_champion(out_root, project, region, display_name, serving_image)


@dsl.container_component
def batch_forecast_op(out_root: str):
    return _cli("dgforecast.predict.batch", out_root)


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def load_forecast_op(out_root: str, project: str, dataset: str):
    """Append the newest forecast file to BigQuery table <dataset>.forecasts."""
    from dgforecast.cloud.load_to_bigquery import load_latest_forecast
    load_latest_forecast(out_root, project, dataset)


@dsl.container_component
def explain_op(out_root: str):
    return _cli("dgforecast.explain.explain", out_root)


@dsl.pipeline(name="dg-demand-forecast-training",
              description="Weekly: validate, build features, train XGBoost, gate, register, forecast.")
def training_pipeline(out_root: str, project: str, region: str, dataset: str,
                      model_display_name: str = "dg-demand-forecast",
                      serving_image: str = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"):
    v = validate_op(out_root=out_root).set_display_name("validate data")
    f = features_op(out_root=out_root).set_display_name("build features").after(v)
    t = train_op(out_root=out_root).set_display_name("train XGBoost").after(f)
    t.set_cpu_limit("8").set_memory_limit("32G")
    e = evaluate_op(out_root=out_root).set_display_name("evaluate and gate").after(t)
    with dsl.If(e.output == "true", name="promoted"):
        u = upload_model_op(out_root=out_root, project=project, region=region,
                            display_name=model_display_name, serving_image=serving_image).set_display_name("register in Vertex")
        b = batch_forecast_op(out_root=out_root).set_display_name("batch forecast").after(u)
        load_forecast_op(out_root=out_root, project=project, dataset=dataset).set_display_name("forecast to BigQuery").after(b)
        explain_op(out_root=out_root).set_display_name("explain").after(u)


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def monitor_op(out_root: str) -> str:
    """Check input drift and outcome decay for the newest forecast. Returns 'true' to retrain."""
    import json
    import logging

    import pandas as pd

    from dgforecast.config import get_paths, load_config, write_json
    from dgforecast.data.generate import CATEGORIES
    from dgforecast.monitor.monitor import input_drift, outcome_decay
    from dgforecast.registry.registry import champion
    cfg = load_config(); paths = get_paths(cfg, out_root)
    fc = max(paths.predictions.glob("forecast_*.parquet")); tag = fc.stem.split("_")[1]
    ch = champion(paths)
    drift = input_drift(pd.read_parquet(paths.models / ch["run_id"] / "reference_sample.parquet"),
                        pd.read_parquet(paths.predictions / f"scoring_features_{tag}.parquet"),
                        cfg, dict(enumerate(CATEGORIES)))
    decay = outcome_decay(pd.read_parquet(fc), pd.read_parquet(paths.raw / "future_actuals.parquet"),
                          pd.read_parquet(paths.raw / "stores.parquet"), cfg, paths)
    retrain = bool(drift["flagged"] or decay["alert"])
    report = {"forecast": tag, "input_drift": drift, "outcome_decay": decay, "retrain": retrain}
    write_json(report, paths.reports / f"monitoring_{tag}.json")
    if retrain:  # a log-based alert policy in Cloud Monitoring watches for this line
        logging.error("DGFORECAST_MODEL_ALERT %s", json.dumps({"forecast": tag, "flagged": drift["flagged"],
                                                              "regions_off": decay["regions_off"]}))
    return "true" if retrain else "false"


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def trigger_retrain_op(project: str, region: str, template_path: str, out_root: str, dataset: str):
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region)
    aiplatform.PipelineJob(display_name="dg-demand-forecast-retrain", template_path=template_path,
                           parameter_values={"out_root": out_root, "project": project,
                                             "region": region, "dataset": dataset}).submit()


@dsl.pipeline(name="dg-demand-forecast-monitoring",
              description="Daily: check drift and accuracy; retrain on alert.")
def monitoring_pipeline(out_root: str, project: str, region: str, dataset: str, training_template_path: str):
    m = monitor_op(out_root=out_root).set_display_name("monitor drift and accuracy")
    with dsl.If(m.output == "true", name="alert"):
        trigger_retrain_op(project=project, region=region, template_path=training_template_path,
                           out_root=out_root, dataset=dataset).set_display_name("start retraining")


def compile_all(out_dir: str = "pipelines") -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    files = [os.path.join(out_dir, "training_pipeline.yaml"), os.path.join(out_dir, "monitoring_pipeline.yaml")]
    compiler.Compiler().compile(training_pipeline, files[0])
    compiler.Compiler().compile(monitoring_pipeline, files[1])
    return files


if __name__ == "__main__":
    for f in compile_all():
        print("compiled", f)
