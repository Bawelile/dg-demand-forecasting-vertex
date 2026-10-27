#!/usr/bin/env bash
# Lesson 11. One-time Google Cloud setup. Run from the repo root after `gcloud auth login`.
# Edit the three values below to match config/config.yaml. Every resource here can incur cost;
# set a budget alert first (Billing > Budgets & alerts) and delete the project when done.
set -euo pipefail

PROJECT_ID="YOUR_GCP_PROJECT_ID"
REGION="us-central1"
BUCKET="YOUR_BUCKET_NAME"          # without gs://
DATASET="dg_forecast"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/ml/dgforecast:latest"

gcloud config set project "$PROJECT_ID"

# 1. Turn on the services we use
gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com storage.googleapis.com \
  artifactregistry.googleapis.com cloudbuild.googleapis.com

# 2. Storage bucket (artifacts + pipeline files) and BigQuery dataset
gcloud storage buckets create "gs://${BUCKET}" --location="$REGION" || true
bq --location=US mk --dataset "${PROJECT_ID}:${DATASET}" || true

# 3. A Docker repository and the project image (built in the cloud, no local Docker needed)
gcloud artifacts repositories create ml --repository-format=docker --location="$REGION" || true
gcloud builds submit --tag "$IMAGE" .

# 4. Generate data locally and copy it to the bucket the pipeline reads
python -m dgforecast.data.generate
gcloud storage cp -r artifacts/data/raw "gs://${BUCKET}/dgforecast/data/"

# 5. Load raw tables into BigQuery and build SQL features (optional path, Lesson 4)
python -m dgforecast.cloud.load_to_bigquery --project "$PROJECT_ID" --dataset "$DATASET"
python -m dgforecast.cloud.bq_features --project "$PROJECT_ID" --dataset "$DATASET"

echo "Setup done. Next: python -m dgforecast.cloud.submit --run"
