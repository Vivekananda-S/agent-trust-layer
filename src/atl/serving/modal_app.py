"""Modal app: Qwen3 8B (agent) and Gemma 4 31B FP8 (customer simulator) on ONE GPU.

Deploy:   modal deploy src/atl/serving/modal_app.py        (ATL_GPU=H100 by default)
Weights:  modal run src/atl/serving/modal_app.py::download_weights   (CPU only, once)
Check:    modal run src/atl/serving/modal_app.py::check_gemma_startup  (cheap L40S, exits)

Cost guards: max_containers=1 (never two GPUs), the container scales to zero 3 minutes after the
last request, and the local GPU ledger (atl.serving.ledger) limits session time. Set a $30
spending limit in the Modal workspace as the hard stop.

Both servers are OpenAI-compatible vLLM servers behind `atl.serving.proxy`, which routes by the
request's `model` ("qwen3-8b" or "gemma-4-31b") and checks the bearer key from the Modal secret
`atl-serving` (ATL_SERVING_KEY). vLLM rejects over-long prompts with an error instead of silently
truncating them, so the Ollama context guard is not needed here.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request

import modal

VLLM_VERSION = "0.30.0"
GPU = os.environ.get("ATL_GPU", "H100")
HF_DIR = "/hf"
CACHE_DIR = "/cache"  # compiled kernels (FA4, Triton, torch.compile) persist across cold starts

# name -> (Hugging Face repo, port, extra vllm args). Gemma starts first and takes 55% of memory;
# Qwen then takes 33% of what remains free. KV caches in FP8 (native on H100) double capacity.
MODELS = {
    "gemma-4-31b": (
        "RedHatAI/gemma-4-31B-it-FP8-dynamic",
        8002,
        [
            "--gpu-memory-utilization",
            "0.55",
            "--reasoning-parser",
            "gemma4",
            "--limit-mm-per-prompt",
            '{"image": 0, "audio": 0}',
        ],
    ),
    "qwen3-8b": (
        "Qwen/Qwen3-8B",
        8001,
        [
            "--gpu-memory-utilization",
            "0.33",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--reasoning-parser",
            "qwen3",
        ],
    ),
}
COMMON_ARGS = ["--max-model-len", "32768", "--kv-cache-dtype", "fp8", "--enable-prefix-caching"]

app = modal.App("atl-serving")
hf_cache = modal.Volume.from_name("atl-hf-cache", create_if_missing=True)
compile_cache = modal.Volume.from_name("atl-compile-cache", create_if_missing=True)
# CUDA *devel* base: Gemma 4's mixed head sizes make vLLM use FlashAttention 4, which compiles
# kernels at start-up and needs nvcc (a slim image crashed with "Could not find nvcc").
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .pip_install(
        f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]", "fastapi", "httpx", "uvicorn"
    )
    .env(
        {
            "HF_HOME": HF_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "CUDA_HOME": "/usr/local/cuda",
            "XDG_CACHE_HOME": CACHE_DIR,
            "VLLM_CACHE_ROOT": f"{CACHE_DIR}/vllm",
            "TRITON_CACHE_DIR": f"{CACHE_DIR}/triton",
        }
    )
    .add_local_python_source("atl")
)


@app.function(image=image, volumes={HF_DIR: hf_cache}, timeout=3600)
def download_weights() -> None:
    """Fill the volume with both models on a CPU container, so no GPU time pays for downloads."""
    from huggingface_hub import snapshot_download

    for repo, _, _ in MODELS.values():
        snapshot_download(repo)
    hf_cache.commit()


def _wait_ready(port: int, proc: subprocess.Popen, timeout_s: float = 1200) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM on port {port} exited with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"vLLM on port {port} not ready after {timeout_s} s")


@app.function(
    image=image,
    gpu="L40S",
    volumes={HF_DIR: hf_cache, CACHE_DIR: compile_cache},
    timeout=1800,
    retries=0,
)
def check_gemma_startup() -> str:
    """Start Gemma alone on a cheaper GPU, answer one request, exit (validates the image)."""
    repo, port, extra = MODELS["gemma-4-31b"]
    extra = [a if a != "0.55" else "0.90" for a in extra]  # alone on the GPU here
    cmd = [
        "vllm",
        "serve",
        repo,
        "--served-model-name",
        "gemma-4-31b",
        "--port",
        str(port),
        "--max-model-len",
        "4096",
        "--kv-cache-dtype",
        "fp8",
        *extra,
    ]
    proc = subprocess.Popen(cmd)
    try:
        _wait_ready(port, proc)
        body = json.dumps(
            {
                "model": "gemma-4-31b",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            }
        )
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=body.encode(),
            headers={"content-type": "application/json"},
        )
        out = json.load(urllib.request.urlopen(req, timeout=300))
        return out["choices"][0]["message"]["content"]
    finally:
        proc.terminate()
        compile_cache.commit()


@app.function(
    image=image,
    gpu=GPU,
    volumes={HF_DIR: hf_cache, CACHE_DIR: compile_cache},
    secrets=[modal.Secret.from_name("atl-serving")],
    max_containers=1,  # never two GPUs
    scaledown_window=180,  # scale to zero 3 minutes after the last request
    timeout=6 * 3600,
)
@modal.concurrent(max_inputs=200)
@modal.web_server(port=8000, startup_timeout=1800)
def serve() -> None:
    """Start Gemma, then Qwen (sequentially, so memory shares are respected), then the proxy."""
    for name, (repo, port, extra) in MODELS.items():
        cmd = [
            "vllm",
            "serve",
            repo,
            "--served-model-name",
            name,
            "--port",
            str(port),
            *COMMON_ARGS,
            *extra,
        ]
        _wait_ready(port, subprocess.Popen(cmd))
    subprocess.Popen(
        [
            "uvicorn",
            "--factory",
            "atl.serving.proxy:app_from_env",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
    )
