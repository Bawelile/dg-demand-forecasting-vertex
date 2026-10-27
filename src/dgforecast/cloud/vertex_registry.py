"""Upload the local champion to Vertex AI Model Registry as a new version.

Vertex keeps every version under one model name. The newest approved version becomes the
"default" version, which is what batch predictions and endpoints use unless told otherwise.
Tested here only by compiling the pipeline; run it in your own Google Cloud project.
"""
from __future__ import annotations

import xgboost as xgb

from dgforecast.config import get_paths, load_config
from dgforecast.registry.registry import champion


def gcs_uri(path: str) -> str:
    """/gcs/my-bucket/a/b (the Vertex mount) -> gs://my-bucket/a/b"""
    return "gs://" + str(path)[len("/gcs/"):] if str(path).startswith("/gcs/") else str(path)


def upload_champion(out_root: str, project: str, region: str, display_name: str, serving_image: str) -> str:
    from google.cloud import aiplatform

    cfg = load_config(); paths = get_paths(cfg, out_root)
    ch = champion(paths)
    folder = paths.registry / ch["version"]
    # Vertex's prebuilt XGBoost serving container loads a file named model.bst
    booster = xgb.Booster(); booster.load_model(folder / "model.json"); booster.save_model(folder / "model.bst")
    aiplatform.init(project=project, location=region)
    existing = aiplatform.Model.list(filter=f'display_name="{display_name}"')
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=gcs_uri(folder),
        serving_container_image_uri=serving_image,
        parent_model=existing[0].resource_name if existing else None,
        is_default_version=True,
        version_aliases=[ch["version"]],
        version_description=f"run {ch['run_id']}, test WAPE {ch['test_wape']:.3f}",
        labels={"run_id": ch["run_id"].lower().replace("_", "-")},
    )
    print("registered", model.resource_name, "version", model.version_id)
    return model.resource_name
