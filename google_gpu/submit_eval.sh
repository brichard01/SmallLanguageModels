#!/usr/bin/env bash
# Evaluate an arbitrary Hugging Face model on the HS-code task on 1x A100 40GB
# (a2-highgpu-1g) via Cloud Batch. Builds the eval image (repo root as context),
# pushes it, ensures a GCS bucket, then submits a job running
# benchmarks/eval_vllm.py. The model is a RUNTIME arg (MODEL=...), so evaluating
# a new checkpoint needs no rebuild -- pass SKIP_BUILD=1 to reuse the image
# already in Artifact Registry and just submit a fresh job.
#
# The job pushes the completions dataset to the Hugging Face Hub, named after the
# model (hscode-eval-<model> under your namespace unless REPO is set). HF_TOKEN
# is injected at runtime from Secret Manager (secret HF_SECRET, default
# "hf-token") so it never lands in the job spec.
#
# Prereq (run once, from YOUR shell where $HF_TOKEN is set):
#   printf '%s' "$HF_TOKEN" | gcloud secrets create hf-token --data-file=- --project=<proj>
#
# Run from the repo root:
#   PROJECT_ID=slm-hscode MODEL=Qwen/Qwen3-4B ./google_gpu/submit_eval.sh
#   # new model, no rebuild:
#   PROJECT_ID=slm-hscode MODEL=Qwen/Qwen3-8B SKIP_BUILD=1 ./google_gpu/submit_eval.sh
#   # keep chain-of-thought in the saved transcripts:
#   PROJECT_ID=slm-hscode MODEL=Qwen/Qwen3-4B KEEP_REASONING=1 ./google_gpu/submit_eval.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
REPO_AR="${REPO_AR:-gpu-demo}"
IMAGE="${IMAGE:-eval-vllm}"
BUCKET="${BUCKET:-${PROJECT_ID}-grpo}"
HF_SECRET="${HF_SECRET:-hf-token}"
LIMIT="${LIMIT:-0}"                 # 0 => whole dataset (no --limit passed)
MODEL="${MODEL:-Qwen/Qwen3-4B}"   # with LORA set, this is the BASE model
LORA="${LORA:-}"                  # HF id of a LoRA adapter to eval on top of MODEL
KEEP_REASONING="${KEEP_REASONING:-0}"
NO_THINKING="${NO_THINKING:-0}"
QUANTIZATION="${QUANTIZATION:-}"    # e.g. awq_marlin for AWQ checkpoints; empty => none
SKIP_BUILD="${SKIP_BUILD:-0}"
OUT="${OUT:-}"                      # GCS backup subdir under /mnt/disks/out (default: model tag)

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "Set PROJECT_ID (env var) or run: gcloud config set project <id>" >&2
  exit 1
fi

# Derive an HF repo + a job name from the run's identity: the adapter id when a
# LoRA is being evaluated, otherwise the (base) model id.
IDENTITY="${LORA:-${MODEL}}"
MODEL_TAG="$(echo "${IDENTITY##*/}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '-' | sed 's/-*$//')"
REPO="${REPO:-hscode-eval-${MODEL_TAG}}"
JOB="${JOB:-eval-${MODEL_TAG}-$(date +%s)}"
OUT="${OUT:-${MODEL_TAG}}"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_AR}/${IMAGE}:latest"
PROJECT_NUM="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

# Batch commands array: expand the two boolean flags to either the flag (with a
# trailing comma, since more args follow) or nothing.
LORA_FLAG="";      [[ -n "${LORA}" ]]                && LORA_FLAG="\"--lora\", \"${LORA}\","
REASONING_FLAG=""; [[ "${KEEP_REASONING}" == "1" ]] && REASONING_FLAG='"--keep-reasoning",'
THINKING_FLAG="";  [[ "${NO_THINKING}" == "1" ]]    && THINKING_FLAG='"--no-thinking",'
QUANT_FLAG="";     [[ -n "${QUANTIZATION}" ]]        && QUANT_FLAG="\"--quantization\", \"${QUANTIZATION}\","

echo ">> Project  : ${PROJECT_ID} (${PROJECT_NUM})"
echo ">> Image    : ${IMAGE_URI}  (SKIP_BUILD=${SKIP_BUILD})"
echo ">> Model    : ${MODEL}"
echo ">> HF repo  : ${REPO}   Limit: ${LIMIT}  keep_reasoning=${KEEP_REASONING} no_thinking=${NO_THINKING}"
echo ">> Bucket   : gs://${BUCKET}/eval-out"
echo ">> Job      : ${JOB}"

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

if [[ "${SKIP_BUILD}" == "1" ]]; then
  echo ">> Skipping build; reusing ${IMAGE_URI}"
else
  ( cd "${ROOT}" && gcloud builds submit \
      --project="${PROJECT_ID}" \
      --config=google_gpu/cloudbuild.eval.yaml \
      --substitutions=_IMAGE_URI="${IMAGE_URI}" \
      . )
fi

# LIMIT=0 => evaluate the whole dataset: drop the "--limit N" pair entirely.
if [[ "${LIMIT}" == "0" ]]; then
  LIMIT_SED='/"--limit", "__LIMIT__",/d'
else
  LIMIT_SED="s#__LIMIT__#${LIMIT}#g"
fi

sed -e "s#__IMAGE_URI__#${IMAGE_URI}#g" \
    -e "s#__BUCKET__#${BUCKET}#g" \
    -e "s#__MODEL__#${MODEL}#g" \
    -e "s#__LORA_FLAG__#${LORA_FLAG}#g" \
    -e "s#__REPO__#${REPO}#g" \
    -e "s#__REASONING_FLAG__#${REASONING_FLAG}#g" \
    -e "s#__THINKING_FLAG__#${THINKING_FLAG}#g" \
    -e "s#__QUANT_FLAG__#${QUANT_FLAG}#g" \
    -e "s#__OUT__#${OUT}#g" \
    -e "${LIMIT_SED}" \
    -e "s#329582102589#${PROJECT_NUM}#g" \
    "${HERE}/batch_job_eval.json" > "/tmp/${JOB}.json"

gcloud batch jobs submit "${JOB}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --config="/tmp/${JOB}.json"

echo
echo ">> Submitted. Watch logs (container stdout lands in batch_task_logs):"
echo "   gcloud logging read 'logName=\"projects/${PROJECT_ID}/logs/batch_task_logs\"' --project=${PROJECT_ID} --freshness=30m --limit=50 --order=desc --format='value(timestamp,textPayload)'"
echo ">> Completions backup: gs://${BUCKET}/eval-out ; and pushed to HF repo ${REPO}"
