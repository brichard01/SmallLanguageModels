# PyTorch + CUDA runtime — already contains torch built against CUDA 12.1.
# The A100 is fully supported by this image.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Unbuffered stdout so print() shows up in Cloud Logging in real time
# (otherwise Python block-buffers and logs appear only at exit).
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY hello_gpu.py .

CMD ["python", "hello_gpu.py"]
