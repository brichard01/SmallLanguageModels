# Hello A100 on Google Cloud Batch

Runs a PyTorch "hello world" on **1x NVIDIA A100 40GB** using the
`a2-highgpu-1g` machine type, via **Cloud Batch** (managed, no VM to babysit).

## Files
| File | Purpose |
|---|---|
| `hello_gpu.py` | Torch script: prints GPU info, runs an 8192² matmul, verifies the result |
| `Dockerfile` | Wraps the script in the official `pytorch/pytorch` CUDA image |
| `batch_job.json` | Batch job spec — `a2-highgpu-1g` + `installGpuDrivers: true` |
| `submit.sh` | One command: build → push → submit → show log commands |

## Quick start
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
