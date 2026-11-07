"""Modal app: one agent model and Gemma 4 31B FP8 (customer simulator) on ONE GPU.

Deploy:   modal deploy src/atl/serving/modal_app.py        (ATL_GPU=H100 by default)
          ATL_AGENT=llama-3.1-8b modal deploy ...        (agent slot; default qwen3-8b)
Weights:  modal run src/atl/serving/modal_app.py::download_weights   (CPU only, once)
Check:    modal run src/atl/serving/modal_app.py::check_startup --model llama-3.1-8b  (L40S)

Cost guards: max_containers=1 (never two GPUs), the container scales to zero 3 minutes after the
last request, and the local GPU ledger (atl.serving.ledger) limits session time. Set a $30
spending limit in the Modal workspace as the hard stop.

Both servers are OpenAI-compatible vLLM servers behind `atl.serving.proxy`, which routes by the
request's `model` ("qwen3-8b", "llama-3.1-8b" or "gemma-4-31b") and checks the bearer key from
the Modal secret `atl-serving` (ATL_SERVING_KEY). vLLM rejects over-long prompts with an error
instead of silently truncating them, so the Ollama context guard is not needed here.
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
# The agent slot holds ONE agent model next to Gemma (the H100 cannot hold Gemma and two 8B
# agents). Chosen at deploy time and passed into the container through a secret (below).
AGENT = os.environ.get("ATL_AGENT", "qwen3-8b")
HF_DIR = "/hf"
# Compiled kernels (FA4, Triton, torch.compile) persist across cold starts. Not /cache: the CUDA
# base image already has that directory, and Modal cannot mount a volume on a non-empty path.
CACHE_DIR = "/atl-compile-cache"

# name -> (Hugging Face repo, port, extra vllm args). Gemma starts first and takes 58% of memory;
# Qwen then takes 33%. KV caches in FP8 (native on H100) double capacity. Context limits differ:
# vLLM refuses to start if max-model-len exceeds the KV cache, and Gemma (31.7 GiB of weights,
# measured on L40S) has little cache left; customer prompts peaked at 3,136 tokens, so 16k is ample.
# Agent prompts reached ~13k tokens, so Qwen keeps 32k.
MODELS = {
    "gemma-4-31b": (
        "RedHatAI/gemma-4-31B-it-FP8-dynamic",
        8002,
        [
            "--max-model-len",
            "16384",
            "--gpu-memory-utilization",
            "0.58",
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
            "--max-model-len",
            "32768",
            "--gpu-memory-utilization",
            "0.33",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--reasoning-parser",
            "qwen3",
        ],
    ),
    # Held-out agent (drift test): same slot, port and context as Qwen, full precision (16 GB).
    # Gated on Hugging Face: the weight download needs a token with the licence accepted.
    "llama-3.1-8b": (
        "meta-llama/Llama-3.1-8B-Instruct",
        8001,
        [
            "--max-model-len",
            "32768",
            "--gpu-memory-utilization",
            "0.33",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "llama3_json",
        ],
    ),
}
CUSTOMER = "gemma-4-31b"


def served_models(agent: str) -> dict[str, tuple[str, int, list[str]]]:
    """The customer plus one agent, customer first (it must claim its memory share first)."""
    if agent == CUSTOMER or agent not in MODELS:
        raise ValueError(
            f"unknown agent model {agent!r}; choose from {sorted(set(MODELS) - {CUSTOMER})}"
        )
    return {CUSTOMER: MODELS[CUSTOMER], agent: MODELS[agent]}


COMMON_ARGS = ["--kv-cache-dtype", "fp8", "--enable-prefix-caching"]

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
        }
    )
    .add_local_python_source("atl")
)


# HF_TOKEN for gated repos (Llama) comes from the Modal secret `huggingface-secret`. Every
# function that starts vLLM needs it too: vLLM queries the Hub for tokenizer files at start-up.
HF_SECRET = modal.Secret.from_name("huggingface-secret")


@app.function(
    image=image,
    volumes={HF_DIR: hf_cache},
    secrets=[HF_SECRET],
    timeout=3600,
)
def download_weights() -> None:
    """Fill the volume with every model on a CPU container, so no GPU time pays for downloads."""
    from huggingface_hub import snapshot_download

    for repo, _, _ in MODELS.values():
        snapshot_download(repo)
    hf_cache.commit()


# Cache locations are set only for the vLLM processes at run time: set image-wide, the image build
# itself wrote into XDG_CACHE_HOME and Modal then refused to mount the volume on that path.
RUNTIME_CACHE_ENV = {
    "XDG_CACHE_HOME": CACHE_DIR,
    "VLLM_CACHE_ROOT": f"{CACHE_DIR}/vllm",
    "TRITON_CACHE_DIR": f"{CACHE_DIR}/triton",
    "TORCHINDUCTOR_CACHE_DIR": f"{CACHE_DIR}/inductor",
}


def _vllm(cmd: list[str]) -> subprocess.Popen:
    """Start a vLLM server with the persistent compile caches."""
    return subprocess.Popen(cmd, env={**os.environ, **RUNTIME_CACHE_ENV})


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


CHECK_TOOL = {
    "type": "function",
    "function": {
        "name": "get_order_details",
        "description": "Get the details of an order.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
}


@app.function(
    image=image,
    gpu="L40S",
    volumes={HF_DIR: hf_cache, CACHE_DIR: compile_cache},
    secrets=[HF_SECRET],
    timeout=1800,
    retries=0,
)
def check_startup(model: str = CUSTOMER) -> str:
    """Start one model alone on a cheaper GPU, answer one request, exit (validates image + access).

    Run before any H100 session that serves a model for the first time: a start-up failure here
    exits once (retries=0) instead of crash-looping an H100.
    """
    repo, port, extra = MODELS[model]
    # Alone on a smaller GPU here: more memory share, small context.
    extra = [
        {"0.58": "0.90", "0.33": "0.90", "16384": "4096", "32768": "4096"}.get(a, a) for a in extra
    ]
    cmd = ["vllm", "serve", repo, "--served-model-name", model, "--port", str(port)]
    proc = _vllm([*cmd, "--kv-cache-dtype", "fp8", *extra])
    try:
        _wait_ready(port, proc)
        request: dict = {
            "model": model,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        }
        if model != CUSTOMER:  # agents must also produce a parsed tool call (tool-call parser)
            request["messages"] = [{"role": "user", "content": "Look up order #W123 for me."}]
            request["tools"] = [CHECK_TOOL]
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(request).encode(),
            headers={"content-type": "application/json"},
        )
        message = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]
        if model == CUSTOMER:
            return message["content"]
        if not message.get("tool_calls"):
            raise RuntimeError(f"{model} returned no parsed tool call: {message}")
        return json.dumps(message["tool_calls"])
    finally:
        proc.terminate()
        compile_cache.commit()


@app.function(
    image=image,
    gpu=GPU,
    volumes={HF_DIR: hf_cache, CACHE_DIR: compile_cache},
    secrets=[
        modal.Secret.from_name("atl-serving"),
        HF_SECRET,
        modal.Secret.from_dict({"ATL_AGENT": AGENT}),
    ],
    max_containers=1,  # never two GPUs
    scaledown_window=180,  # scale to zero 3 minutes after the last request
    timeout=6 * 3600,
)
@modal.concurrent(max_inputs=200)
@modal.web_server(port=8000, startup_timeout=1800)
def serve() -> None:
    """Start Gemma, then the agent (in order, so memory shares are respected), then the proxy."""
    for name, (repo, port, extra) in served_models(os.environ["ATL_AGENT"]).items():
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
        _wait_ready(port, _vllm(cmd))
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
