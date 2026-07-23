# A100 jobs on Google Cloud Batch

Infra for running GPU jobs on **1x NVIDIA A100 40GB** (`a2-highgpu-1g`) via
**Cloud Batch** (managed, no VM to babysit). Two jobs live here:

- **Hello-world** — a cheap (~2 min, ~$0.25) sanity check that the A100 path
  works end to end.
- **GRPO training** — builds the project image and runs
  `training_methods/grpo_hscode.py` (GRPO/QLoRA fine-tuning of Qwen3-4B on the
  HS-code env reward), writing the trained adapter to a GCS bucket.
- **Model evaluation** — builds a vLLM image and runs `benchmarks/eval_vllm.py`
  to benchmark **any Hugging Face model** on the HS-code task. The model is a
  runtime arg, so a new checkpoint needs no rebuild; completions are pushed to
  the Hub named after the model.

## Files
| File | Purpose |
|---|---|
| **Hello-world** | |
| `hello_gpu.py` | Torch script: prints GPU info, runs an 8192² matmul, verifies the result |
| `Dockerfile` | Wraps the script in the official `pytorch/pytorch` CUDA image |
| `batch_job.json` | Batch job spec — `a2-highgpu-1g` + `installGpuDrivers: true` |
| `submit.sh` | One command: build → push → submit → show log commands |
| **GRPO training** | |
| `Dockerfile.grpo` | Project image: torch 2.6/cu124 + transformers/peft/trl(main)/bitsandbytes, bundles `hscode_env.py`, `training_methods/grpo_hscode.py`, `data/` |
| `cloudbuild.grpo.yaml` | Builds `Dockerfile.grpo` with the **repo root** as context |
| `batch_job_grpo.json` | Batch job spec — A100, 150 GB disk, 3 h cap, GCS output volume at `/mnt/disks/out` |
| `submit_grpo.sh` | One command: build → push → ensure bucket → submit → show log commands |
| **Model evaluation** | |
| `Dockerfile.eval` | vLLM image (`vllm/vllm-openai` + `datasets`); bundles `hscode_env.py`, `benchmarks/eval_vllm.py`, `data/`. Model NOT baked in — passed at runtime |
| `cloudbuild.eval.yaml` | Builds `Dockerfile.eval` with the **repo root** as context |
| `batch_job_eval.json` | Batch job spec — A100, GCS output at `/mnt/disks/out`; `__MODEL__` / flag placeholders filled by the submit script |
| `submit_eval.sh` | One command: build → push → ensure bucket → submit. `MODEL=` picks the checkpoint; `SKIP_BUILD=1` reuses the image for a new model |

## Model evaluation quick start
```bash
# run from the repo root
PROJECT_ID=<your-project> MODEL=Qwen/Qwen3-4B ./google_gpu/submit_eval.sh
```
Builds the vLLM image, runs the agentic HS-code loop over the dataset on an A100,
and pushes the per-episode **completions** (full transcript + submitted code,
reward, correct) to the Hub as `hscode-eval-<model>` under your namespace, with a
backup at `gs://<PROJECT_ID>-grpo/eval-out`. Knobs (env vars):
- `MODEL` — any HF model id (default `Qwen/Qwen3-4B`).
- `LORA` — HF id of a LoRA adapter to evaluate on top of `MODEL` (the base model
  and rank are read from the adapter config; the adapter names the output repo).
  Use this to benchmark a distill/GRPO fine-tune, e.g.
  `LORA=brichard01/qwen3-4b-hscode-distill`.
- `SKIP_BUILD=1` — **evaluate a new model without rebuilding** the image.
- `KEEP_REASONING=1` — keep `<think>` chain-of-thought in the saved transcripts
  and re-fed context (default: strip it).
- `NO_THINKING=1` — disable the model's thinking mode entirely.
- `LIMIT=N` — first N rows only (`0` = whole dataset, the default).
- `REPO=<name-or-namespace/name>` — override the HF dataset repo.
- `QUANTIZATION=awq_marlin` — vLLM quantization for AWQ (or other) checkpoints.

HF_TOKEN is injected at runtime from Secret Manager (secret `hf-token`); create
it once (see the note in `submit_distill.sh`). Watch logs the same way as GRPO
below.

## GRPO quick start
```bash
# run from the repo root
PROJECT_ID=<your-project> ./google_gpu/submit_grpo.sh
```
Builds the image, submits the A100 job, and writes the LoRA adapter to
`gs://<PROJECT_ID>-grpo/grpo-out`. Override defaults via env vars: `REGION`,
`REPO`, `IMAGE`, `JOB`, `BUCKET`. It trains on `data/benchmark_dataset.csv` by
default (`GRPO_DATA` in the job spec); point that at the full dataset for a real
run. Watch it with (note: container stdout lands in `batch_task_logs`, and Batch
leaves the `job_id` label blank on those lines, so filter by log name):
```bash
gcloud logging read 'logName="projects/<PROJECT_ID>/logs/batch_task_logs"' \
  --project=<PROJECT_ID> --freshness=30m --limit=50 --order=desc \
  --format="value(timestamp,textPayload)"
```

## Hello-world quick start
```bash
export PROJECT_ID=$(gcloud config get-value project)   # or set it explicitly
cd google_gpu
./submit.sh
```
That will:
1. Create an Artifact Registry repo `gpu-demo` (once).
2. Build & push the image with **Cloud Build** (no local Docker/GPU needed).
3. Submit the Batch job. Batch provisions the A100 VM, installs GPU drivers,
   runs the container, then tears the VM down.

Override defaults via env vars: `REGION` (default `us-central1`), `REPO`, `IMAGE`, `JOB`.

## Watching it
```bash
# Status: QUEUED → SCHEDULED → RUNNING → SUCCEEDED  (A100 capacity can take a few min)
gcloud batch jobs describe <JOB> --location=us-central1

# The actual GPU output (torch version, "Tesla A100", TFLOP/s, "Hello from the A100")
gcloud logging read 'labels."batch.googleapis.com/job_id"="<JOB>"' \
  --limit=50 --order=asc --format='value(textPayload)'
```

## How it works / why each piece
- **`a2-highgpu-1g`** = 12 vCPU, 85 GB RAM, **1× A100 40GB**. For `a2` machine
  types the GPU is *built into* the machine type, so `batch_job.json` does **not**
  list accelerators separately (unlike `n1` + `nvidia-tesla-t4`).
- **`installGpuDrivers: true`** tells Batch to install the NVIDIA driver on the
  host before the container starts — this is what makes `torch.cuda.is_available()`
  return `True`.
- **`--privileged`** container option gives the container access to the GPU
  device nodes. (You can also mount `/dev/nvidia*` explicitly, but privileged is
  simplest for a demo.)
- **`computeResource`** is set just under the machine's limits (12000 mCPU,
  ~82 GiB) so the single task gets the whole node and its GPU.
- **Cloud Build** builds the image server-side, so you don't need Docker or a
  local GPU on your Mac.

## Cost & cleanup
- `a2-highgpu-1g` is ~**$3.7/hr** on-demand. This job runs a couple of minutes,
  and Batch deletes the VM automatically when the job finishes.
- To run cheaper, set `"provisioningModel": "SPOT"` in `batch_job.json`
  (can be preempted, ~60–70% cheaper).
- Delete finished jobs: `gcloud batch jobs delete <JOB> --location=us-central1`.

## Prereqs (you've already done step 1)
```bash
# 1. APIs (done)
gcloud services enable batch.googleapis.com compute.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
# 2. A100 quota: ensure "NVIDIA A100 GPUs" quota >= 1 in your region
#    (Console → IAM & Admin → Quotas). New projects often start at 0.
```

## Troubleshooting
- **Job stuck in QUEUED/SCHEDULED** → usually A100 capacity or quota. Try another
  zone/region (e.g. `REGION=us-east1`) or request quota.
- **`CUDA is NOT available`** in logs → drivers didn't install; confirm
  `installGpuDrivers: true` and that the machine type is an `a2`.
- **PERMISSION_DENIED on submit** → the Batch/Compute default service account
  needs `roles/batch.jobsEditor` and access to Artifact Registry
  (`roles/artifactregistry.reader`).
