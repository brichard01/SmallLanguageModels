"""Hello-world PyTorch job that proves an NVIDIA A100 (40 GB) is visible and usable.

Runs inside a Cloud Batch container on an `a2-highgpu-1g` VM (1x A100 40GB).
It prints the environment, does a real matmul on the GPU, and verifies the
result against a CPU reference so we know the device actually computed.
"""

import torch


def main() -> None:
    print("=" * 60)
    print(f"torch version      : {torch.__version__}")
    print(f"CUDA available     : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is NOT available — the GPU/driver is not visible to this "
            "process. Check that the VM is an a2 machine and that GPU drivers "
            "were installed (Batch: installGpuDrivers=true)."
        )

    print(f"CUDA runtime       : {torch.version.cuda}")
    print(f"device count       : {torch.cuda.device_count()}")

    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"device 0           : {name}")
    print(f"total memory       : {total_gb:.1f} GiB")
    print("=" * 60)

    # A real workload on the GPU: 8192x8192 matmul in fp32.
    torch.manual_seed(0)
    n = 8192
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c = a @ b
    end.record()
    torch.cuda.synchronize()

    elapsed_s = start.elapsed_time(end) / 1000.0
    flops = 2 * n**3  # multiply-add per output element
    tflops = flops / elapsed_s / 1e12
    print(f"matmul {n}x{n}   : {elapsed_s * 1000:.1f} ms  (~{tflops:.1f} TFLOP/s)")

    # Verify a small slice against CPU so we trust the GPU result.
    ref = (a[:2].cpu() @ b.cpu())
    ok = torch.allclose(c[:2].cpu(), ref, rtol=1e-3, atol=1e-3)
    print(f"result verified    : {ok}")

    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"peak GPU memory    : {peak_gb:.2f} GiB")
    print("Hello from the A100 🎉")

    if not ok:
        raise SystemExit("GPU result did not match CPU reference!")


if __name__ == "__main__":
    main()
