#!/usr/bin/env bash
# Build the distillation image (repo root as context), push to Artifact Registry,
# ensure a GCS bucket exists, then submit a Cloud Batch job that runs
# training_methods/distill_generate.py on 1x A100 40GB (a2-highgpu-1g). The job
# generates teacher (Qwen3-32B-AWQ) trajectories + top-k logprobs and pushes the
# dataset to the Hugging Face Hub. HF_TOKEN is injected at runtime from Google
# Secret Manager (secret name: HF_SECRET, default "hf-token") so it never lands
# in the job spec.
#
# Prereq (run once, from YOUR shell where $HF_TOKEN is set — keeps the token out
# of any command log or job spec):
#   printf '%s' "$HF_TOKEN" | gcloud secrets create hf-token --data-file=- --project=<proj>
#
# Run from the repo root:
#   PROJECT_ID=slm-hscode LIMIT=20 REPO=hscode-distill-qwen3-32b ./google_gpu/submit_distill.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
REPO_AR="${REPO_AR:-gpu-demo}"
IMAGE="${IMAGE:-distill-qwen3}"
JOB="${JOB:-distill-qwen3-$(date +%s)}"
BUCKET="${BUCKET:-${PROJECT_ID}-grpo}"
HF_SECRET="${HF_SECRET:-hf-token}"
LIMIT="${LIMIT:-20}"
REPO="${REPO:-hscode-distill-qwen3-32b}"   # HF dataset repo (name-only -> your namespace)

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "Set PROJECT_ID (env var) or run: gcloud config set project <id>" >&2
  exit 1
fi

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_AR}/${IMAGE}:latest"
PROJECT_NUM="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

echo ">> Project : ${PROJECT_ID} (${PROJECT_NUM})"
echo ">> Image   : ${IMAGE_URI}"
echo ">> Bucket  : gs://${BUCKET}/distill-out"
echo ">> Secret  : ${HF_SECRET}"
echo ">> HF repo : ${REPO}   Limit: ${LIMIT}"
echo ">> Job     : ${JOB}"

# Fail early with a clear message if the secret isn't there yet.
if ! gcloud secrets describe "${HF_SECRET}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "ERROR: secret '${HF_SECRET}' not found in ${PROJECT_ID}. Create it first (from your shell):" >&2
  echo "  printf '%s' \"\$HF_TOKEN\" | gcloud secrets create ${HF_SECRET} --data-file=- --project=${PROJECT_ID}" >&2
  exit 1
fi

gcloud artifacts repositories create "${REPO_AR}" \
  --repository-format=docker --location="${REGION}" \
  --description="GPU images" --project="${PROJECT_ID}" 2>/dev/null || true
gcloud storage buckets create "gs://${BUCKET}" \
  --location="${REGION}" --project="${PROJECT_ID}" 2>/dev/null || true

( cd "${ROOT}" && gcloud builds submit \
    --project="${PROJECT_ID}" \
    --config=google_gpu/cloudbuild.distill.yaml \
    --substitutions=_IMAGE_URI="${IMAGE_URI}" \
    . )

sed -e "s#__IMAGE_URI__#${IMAGE_URI}#g" \
    -e "s#__BUCKET__#${BUCKET}#g" \
    -e "s#__LIMIT__#${LIMIT}#g" \
    -e "s#__REPO__#${REPO}#g" \
    -e "s#329582102589#${PROJECT_NUM}#g" \
    "${HERE}/batch_job_distill.json" > "/tmp/${JOB}.json"
gcloud batch jobs submit "${JOB}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --config="/tmp/${JOB}.json"

echo
echo ">> Submitted. Watch logs (container stdout lands in batch_task_logs):"
echo "   gcloud logging read 'logName=\"projects/${PROJECT_ID}/logs/batch_task_logs\"' --project=${PROJECT_ID} --freshness=30m --limit=50 --order=desc --format='value(timestamp,textPayload)'"
echo ">> Dataset backup: gs://${BUCKET}/distill-out ; and pushed to HF repo ${REPO}"
