"""Lesson 11. Compile, upload, run, and schedule the pipelines on Vertex AI.

    python -m dgforecast.cloud.submit --run            # run the training pipeline once
    python -m dgforecast.cloud.submit --schedule       # weekly training + daily monitoring

Settings come from config/config.yaml (project, region, bucket, dataset, image).
Written against the google-cloud-aiplatform SDK; check names against the current docs
if a call fails, since Google renamed Vertex AI to Gemini Enterprise Agent Platform in 2026.
"""
from __future__ import annotations

import argparse

from dgforecast.config import load_config
from dgforecast.pipeline.vertex_pipeline import compile_all


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true"); ap.add_argument("--schedule", action="store_true")
    a = ap.parse_args(argv)
    from google.cloud import aiplatform, storage

    p = load_config()["project"]
    project, region, bucket, dataset = p["gcp_project"], p["region"], p["bucket"], p["bq_dataset"]
    bucket_name = bucket.replace("gs://", "").strip("/")
    out_root = f"/gcs/{bucket_name}/dgforecast"
    pipeline_root = f"{bucket}/pipeline_root"

    # upload compiled templates so scheduled and retrain runs can find them
    train_json, monitor_json = compile_all()
    blob_bucket = storage.Client(project=project).bucket(bucket_name)
    uris = {}
    for local in (train_json, monitor_json):
        name = f"dgforecast/pipelines/{local.split('/')[-1]}"
        blob_bucket.blob(name).upload_from_filename(local)
        uris[local] = f"gs://{bucket_name}/{name}"
    aiplatform.init(project=project, location=region, staging_bucket=bucket)

    params = {"out_root": out_root, "project": project, "region": region, "dataset": dataset}
    train_job = aiplatform.PipelineJob(display_name="dg-demand-forecast-training",
                                       template_path=uris[train_json], pipeline_root=pipeline_root,
                                       parameter_values=params, enable_caching=False)
    if a.run:
        train_job.submit()
        print("submitted training run:", train_job.resource_name)
    if a.schedule:
        train_job.create_schedule(display_name="dg-forecast-weekly-train",
                                  cron="TZ=America/New_York 0 5 * * 1")  # Mondays 5:00 am
        monitor_job = aiplatform.PipelineJob(
            display_name="dg-demand-forecast-monitoring", template_path=uris[monitor_json],
            pipeline_root=pipeline_root, enable_caching=False,
            parameter_values={**params, "training_template_path": uris[train_json]})
        monitor_job.create_schedule(display_name="dg-forecast-daily-monitor",
                                    cron="TZ=America/New_York 0 7 * * *")  # daily 7:00 am
        print("scheduled weekly training and daily monitoring")


if __name__ == "__main__":
    main()
