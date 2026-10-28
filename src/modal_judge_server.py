"""Qwen2.5-32B-Instruct-AWQ behind an OpenAI-compatible vLLM server on Modal.

Purpose: score the existing v2 runs (data/runs) with a stronger judge than the two used in
the report (Qwen 2.5 14B, Gemma 3 12B). Same frozen v2 judge prompt, no new generation.

Pattern follows the standard Modal vLLM web-server recipe.

Deploy (token embedded at deploy time, never printed):
    export MODAL_JUDGE_TOKEN=$(python3 -c "import secrets;print(secrets.token_urlsafe(24))")
    modal deploy src/modal_judge_server.py

Stop (billing):
    modal app stop m3-strong-judge

Smoke test:
    curl -s -H "Authorization: Bearer $MODAL_JUDGE_TOKEN" https://<url>/healthz
"""
import os
import subprocess
import time

import modal

PORT = 8000
MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
GPU_MEM = 0.90
HF_VOLUME = "pgg-swap-hf"  # reuse: already holds this model from the collusion sprint

hf_volume = modal.Volume.from_name(HF_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.2", "transformers==4.56.2", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)

app = modal.App("m3-strong-judge")


@app.function(
    image=image,
    gpu="L40S",                     # 48 GB; 19 GB of AWQ weights fits comfortably
    volumes={"/root/.cache/huggingface": hf_volume},
    timeout=3600,
    scaledown_window=180,           # stop billing ~3 min after the last request
    max_containers=1,
)
@modal.web_server(port=PORT, startup_timeout=900)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--served-model-name", MODEL,
        "--max-model-len", "8192",
        "--gpu-memory-utilization", str(GPU_MEM),
        "--max-num-seqs", "32",
        "--disable-log-requests",
    ]
    print("launching:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)


@app.function(image=image, volumes={"/root/.cache/huggingface": hf_volume}, timeout=1800)
def download():
    """Run once on CPU to avoid a slow first web request: modal run ...::download"""
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download(MODEL, ignore_patterns=["original/*", "*.pth"])
    hf_volume.commit()
    print(f"downloaded {MODEL} in {time.time() - t0:.0f}s", flush=True)
