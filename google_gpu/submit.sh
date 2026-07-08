#!/usr/bin/env bash
# End-to-end: build the image, push to Artifact Registry, submit a Cloud Batch
# job that runs hello_gpu.py on 1x A100 40GB (a2-highgpu-1g), and tail the logs.
#
# Usage:
#   PROJECT_ID=my-proj ./submit.sh
# Optional overrides (defaults shown):
#   REGION=us-central1  REPO=gpu-demo  IMAGE=hello-gpu  JOB=hello-a100
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-gpu-demo}"
IMAGE="${IMAGE:-hello-gpu}"
JOB="${JOB:-hello-a100-$(date +%s)}"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "Set PROJECT_ID (env var) or run: gcloud config set project <id>" >&2
  exit 1
fi

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest"
echo ">> Project : ${PROJECT_ID}"
echo ">> Region  : ${REGION}"
echo ">> Image   : ${IMAGE_URI}"
echo ">> Job     : ${JOB}"

# 1. Artifact Registry repo (create once; ignore error if it already exists).
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker --location="${REGION}" \
  --description="GPU demo images" 2>/dev/null || true

# 2. Build + push with Cloud Build (no local Docker/nvidia needed).
gcloud builds submit --tag "${IMAGE_URI}" .

# 3. Render the job spec with the real image URI, then submit.
sed "s#__IMAGE_URI__#${IMAGE_URI}#g" batch_job.json > /tmp/${JOB}.json
gcloud batch jobs submit "${JOB}" \
  --location="${REGION}" \
  --config=/tmp/${JOB}.json

echo
echo ">> Submitted. Watch status with:"
echo "   gcloud batch jobs describe ${JOB} --location=${REGION}"
echo ">> Read the GPU output once the job SUCCEEDED with:"
echo "   gcloud logging read 'labels.\"batch.googleapis.com/job_id\"=\"${JOB}\"' --project=${PROJECT_ID} --limit=50 --order=asc --format='value(textPayload)'"
