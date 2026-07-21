#!/usr/bin/env bash
# End-to-end: build a GRPO training image (repo root as build context), push to
# Artifact Registry, ensure a GCS output bucket exists, then submit a Cloud Batch
# job that runs grpo_hscode.py on 1x A100 40GB (a2-highgpu-1g) and writes the
# trained LoRA adapter to gs://$BUCKET/grpo-out.
#
# Run from the repo root:
#   PROJECT_ID=slm-hscode ./google_gpu/submit_grpo.sh
# Optional overrides (defaults shown):
#   REGION=us-central1  REPO=gpu-demo  IMAGE=grpo-hscode  JOB=grpo-hscode-<ts>
#   BUCKET=<PROJECT_ID>-grpo
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-gpu-demo}"
IMAGE="${IMAGE:-grpo-hscode}"
JOB="${JOB:-grpo-hscode-$(date +%s)}"
BUCKET="${BUCKET:-${PROJECT_ID}-grpo}"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "Set PROJECT_ID (env var) or run: gcloud config set project <id>" >&2
  exit 1
fi

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

echo ">> Project : ${PROJECT_ID}"
echo ">> Region  : ${REGION}"
echo ">> Image   : ${IMAGE_URI}"
echo ">> Bucket  : gs://${BUCKET}/grpo-out"
echo ">> Job     : ${JOB}"

# 1. Artifact Registry repo (create once; ignore error if it already exists).
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker --location="${REGION}" \
  --description="GPU images" --project="${PROJECT_ID}" 2>/dev/null || true

# 2. GCS output bucket (create once). The Batch task mounts it at /mnt/out.
gcloud storage buckets create "gs://${BUCKET}" \
  --location="${REGION}" --project="${PROJECT_ID}" 2>/dev/null || true

# 3. Build + push with Cloud Build, using the repo root as build context so the
#    image can COPY grpo_hscode.py, hscode_env.py and data/.
( cd "${ROOT}" && gcloud builds submit \
    --project="${PROJECT_ID}" \
    --config=google_gpu/cloudbuild.grpo.yaml \
    --substitutions=_IMAGE_URI="${IMAGE_URI}" \
    . )

# 4. Render the job spec with the real image URI + bucket, then submit.
sed -e "s#__IMAGE_URI__#${IMAGE_URI}#g" \
    -e "s#__BUCKET__#${BUCKET}#g" \
    "${HERE}/batch_job_grpo.json" > "/tmp/${JOB}.json"
gcloud batch jobs submit "${JOB}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --config="/tmp/${JOB}.json"

echo
echo ">> Submitted. Watch status with:"
echo "   gcloud batch jobs describe ${JOB} --location=${REGION} --project=${PROJECT_ID}"
echo ">> Stream training logs with:"
echo "   gcloud logging read 'labels.\"batch.googleapis.com/job_id\"=\"${JOB}\"' --project=${PROJECT_ID} --limit=100 --order=asc --format='value(textPayload)'"
echo ">> Trained adapter will land in: gs://${BUCKET}/grpo-out"
