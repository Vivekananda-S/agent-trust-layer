"""Tiny authenticated router in front of two vLLM servers sharing one GPU.

Modal exposes one port per web endpoint; the agent model and the customer simulator each run
their own vLLM server, so this proxy routes OpenAI-style requests by their `model` field and
checks a bearer key. Routing and auth are pure functions (tested locally); the HTTP app is
built only inside the Modal container, where fastapi and httpx are installed.

No `from __future__ import annotations` here: FastAPI resolves handler type hints at runtime, and
`Request` is imported inside `build_app`, so string annotations made it a missing query field.
"""

import hmac
from typing import Any


def route(model: str, upstreams: dict[str, str]) -> str | None:
    """Base URL of the vLLM server that serves `model`, or None if unknown."""
    return upstreams.get(model)


def authorized(header: str | None, key: str) -> bool:
    """Constant-time check of an `Authorization: Bearer <key>` header."""
    if not key or not header or not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header.removeprefix("Bearer "), key)


def build_app(upstreams: dict[str, str], key: str) -> Any:
    """FastAPI app that forwards /v1/* calls to the upstream for the requested model."""
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response

    app = FastAPI()
    client = httpx.AsyncClient(timeout=600)

    @app.get("/health")
    async def health() -> dict[str, list[str]]:
        return {"models": sorted(upstreams)}

    @app.post("/v1/{path:path}")
    async def forward(path: str, request: Request) -> Response:
        if not authorized(request.headers.get("authorization"), key):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.body()
        model = (await request.json()).get("model", "")
        base = route(model, upstreams)
        if base is None:
            return JSONResponse({"error": f"unknown model {model!r}"}, status_code=404)
        upstream = await client.post(
            f"{base}/v1/{path}", content=body, headers={"content-type": "application/json"}
        )
        return Response(upstream.content, upstream.status_code, media_type="application/json")

    return app


UPSTREAMS = {"qwen3-8b": "http://127.0.0.1:8001", "gemma-4-31b": "http://127.0.0.1:8002"}


def app_from_env() -> Any:
    """uvicorn factory used inside the Modal container (key from the `atl-serving` secret)."""
    import os

    return build_app(UPSTREAMS, os.environ["ATL_SERVING_KEY"])
