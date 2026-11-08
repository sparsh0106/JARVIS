"""FastAPI Application Lifecycle and Routes (/health, /api/chat, /api/jarvis/stream)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from ..agent import JarvisAgent
from ..config import DEFAULT_GGUF_PATH, JarvisConfig, LLMConfig, MemoryConfig
from .llm_service import LocalLLMService

logger = logging.getLogger("jarvis.api.server")

MODEL_PATH = os.getenv("JARVIS_GGUF_PATH", DEFAULT_GGUF_PATH)
N_CTX = int(os.getenv("JARVIS_N_CTX", "8192"))
N_GPU_LAYERS = int(os.getenv("JARVIS_N_GPU_LAYERS", "-1"))
MEMORY_DIR = os.getenv("JARVIS_MEMORY_DIR", "jarvis_data/memory")

llm_service = LocalLLMService(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_gpu_layers=N_GPU_LAYERS,
)
_agent: Optional[JarvisAgent] = None


def _init_runtime(custom_agent: Optional[JarvisAgent] = None) -> JarvisAgent:
    """Initialize and bind a single shared JarvisAgent and LocalLLMService instance."""
    global _agent
    if custom_agent is not None:
        _agent = custom_agent
        llm_service.bind_engine(custom_agent.llm_engine)
        return _agent

    if _agent is None:
        cfg = JarvisConfig(
            memory=MemoryConfig(base_dir=Path(MEMORY_DIR)),
            llm=LLMConfig(
                model_path=MODEL_PATH,
                n_ctx=N_CTX,
                n_gpu_layers=N_GPU_LAYERS,
            ),
        )
        _agent = JarvisAgent(config=cfg)
        llm_service.bind_engine(_agent.llm_engine)
    return _agent


def _validate_agent_stream_payload(payload: Any) -> Dict[str, Any]:
    """Validate incoming request body for /api/jarvis/stream."""
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400, detail="JSON body must be an object"
        )

    user_text = payload.get("user_text") or payload.get("message")
    history = payload.get("history", [])

    # Also support OpenAI-style `messages` array if passed to /api/jarvis/stream
    if user_text is None and isinstance(payload.get("messages"), list):
        msgs = payload["messages"]
        if not msgs or not isinstance(msgs[-1], dict):
            raise HTTPException(
                status_code=422, detail="'messages' must be a non-empty list"
            )
        user_text = msgs[-1].get("content")
        history = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in msgs[:-1]
            if isinstance(m, dict) and m.get("role") in {"user", "assistant"}
        ]

    if not isinstance(user_text, str) or not user_text.strip():
        raise HTTPException(
            status_code=422,
            detail="'user_text' (or 'message') must be a non-empty string",
        )

    if not isinstance(history, list):
        raise HTTPException(
            status_code=422, detail="'history' must be a list of message objects"
        )

    validated_history: List[Dict[str, str]] = []
    for idx, item in enumerate(history):
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=422, detail=f"history[{idx}] must be an object"
            )
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise HTTPException(
                status_code=422,
                detail=f"history[{idx}] must have role ('user'|'assistant') and string content",
            )
        validated_history.append({"role": role, "content": content})

    max_tool_rounds = payload.get("max_tool_rounds")
    if max_tool_rounds is not None:
        if (
            not isinstance(max_tool_rounds, int)
            or isinstance(max_tool_rounds, bool)
            or max_tool_rounds < 1
            or max_tool_rounds > 12
        ):
            raise HTTPException(
                status_code=422,
                detail="'max_tool_rounds' must be an integer between 1 and 12",
            )

    return {
        "user_text": user_text.strip(),
        "history": validated_history,
        "max_tool_rounds": max_tool_rounds,
    }


def create_app(custom_agent: Optional[JarvisAgent] = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        agent = await run_in_threadpool(_init_runtime, custom_agent)
        application.state.agent = agent
        application.state.llm_service = llm_service
        yield
        if _agent is not None:
            await run_in_threadpool(_agent.memory_store.flush)

    application = FastAPI(title="JARVIS Local LLM API", lifespan=lifespan)

    @application.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": llm_service._llm is not None,
            "memory_count": _agent.memory_store.size if _agent is not None else 0,
        }

    @application.post("/api/chat")
    async def chat_endpoint(request: Request):
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="Invalid JSON request body"
            ) from exc

        return await llm_service.chat(payload)

    @application.post("/api/jarvis/stream")
    async def jarvis_stream_endpoint(request: Request) -> StreamingResponse:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="Invalid JSON request body"
            ) from exc

        params = _validate_agent_stream_payload(payload)
        if _agent is None:
            raise HTTPException(
                status_code=503, detail="JARVIS agent is not initialized"
            )

        req_id = uuid.uuid4().hex[:12]

        async def event_generator():
            try:
                async for event in _agent.stream_turn(
                    user_text=params["user_text"],
                    history=params["history"],
                    max_tool_rounds=params["max_tool_rounds"],
                    request_id=req_id,
                ):
                    yield (
                        json.dumps(
                            event,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            except asyncio.CancelledError:
                logger.info(
                    "Client disconnected during /api/jarvis/stream (req_id=%s)",
                    req_id,
                )
                raise
            except Exception:
                logger.exception(
                    "Unhandled error in /api/jarvis/stream (req_id=%s)", req_id
                )
                yield (
                    json.dumps(
                        {
                            "event": "error",
                            "code": "generation_failed",
                            "message": "The model could not complete this response.",
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                yield json.dumps({"event": "done"}, separators=(",", ":")) + "\n"

        return StreamingResponse(
            event_generator(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return application


app = create_app()
