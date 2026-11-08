"""Raw LLM Streaming Service for FastAPI (/api/chat) with Threadpool Offloading & Lock Serialization."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from llama_cpp import Llama
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from ..config import LLMConfig
from ..llm import LocalLLMEngine

logger = logging.getLogger("jarvis.llm_service")

_ALLOWED_CHAT_KEYS = {
    "messages",
    "stream",
    "max_tokens",
    "temperature",
    "top_p",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "repeat_penalty",
    "seed",
    "model",
}


class LocalLLMService:
    """
    Provides validated synchronous and NDJSON-streaming chat completions over a single
    GGUF model instance, sharing `LocalLLMEngine` and its `asyncio.Lock` with `JarvisAgent`.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        max_tokens_limit: int = 2048,
        engine: Optional[LocalLLMEngine] = None,
    ) -> None:
        self._engine: Optional[LocalLLMEngine] = engine
        cfg = engine.config if engine is not None else LLMConfig()
        self.model_path = model_path or cfg.model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens_limit = max_tokens_limit
        self._fallback_lock = asyncio.Lock()

    @property
    def _llm(self) -> Optional[Llama]:
        return self._engine.llm if self._engine is not None else None

    @property
    def _generation_lock(self) -> asyncio.Lock:
        if self._engine is not None:
            return self._engine.generation_lock
        return self._fallback_lock

    def bind_engine(self, engine: LocalLLMEngine) -> None:
        """Bind an already-initialized LocalLLMEngine so /api/chat and /api/jarvis/stream share one model."""
        self._engine = engine
        self.model_path = engine.config.model_path
        self.n_ctx = engine.config.n_ctx
        self.n_gpu_layers = engine.config.n_gpu_layers
        self.max_tokens_limit = engine.config.max_tokens_limit

    def load(self) -> None:
        """Load the GGUF model once during application startup if not already bound."""
        if self._engine is not None:
            return

        cfg = LLMConfig(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            max_tokens_limit=self.max_tokens_limit,
        )
        self._engine = LocalLLMEngine(config=cfg)
        logger.info("Loaded GGUF model from %s", self.model_path)

    def _get_engine(self) -> LocalLLMEngine:
        if self._engine is None or self._engine.llm is None:
            raise RuntimeError("LLM has not been loaded")
        return self._engine

    def _validate_payload(self, payload: Any) -> Dict[str, Any]:
        """Validate incoming /api/chat JSON payload and enforce parameter bounds."""
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400, detail="JSON body must be an object"
            )

        unknown_keys = set(payload.keys()) - _ALLOWED_CHAT_KEYS
        if unknown_keys:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported request fields: {sorted(unknown_keys)}",
            )

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(
                status_code=422, detail="'messages' must be a non-empty list"
            )

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise HTTPException(
                    status_code=422, detail=f"messages[{index}] must be an object"
                )
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise HTTPException(
                    status_code=422, detail=f"messages[{index}] has an invalid role"
                )
            if "content" not in message and "tool_calls" not in message:
                raise HTTPException(
                    status_code=422, detail=f"messages[{index}] has no content"
                )
            if "content" in message and not isinstance(
                message["content"], (str, type(None))
            ):
                raise HTTPException(
                    status_code=422,
                    detail=f"messages[{index}].content must be a string",
                )

        stream = payload.get("stream", True)
        if not isinstance(stream, bool):
            raise HTTPException(
                status_code=422, detail="'stream' must be a boolean"
            )

        # Strip compatibility-only model selector; server owns model choice.
        params = dict(payload)
        params.pop("model", None)
        params["stream"] = stream

        max_tokens = params.get("max_tokens")
        if max_tokens is not None:
            if (
                not isinstance(max_tokens, int)
                or isinstance(max_tokens, bool)
                or max_tokens < 1
            ):
                raise HTTPException(
                    status_code=422,
                    detail="'max_tokens' must be a positive integer",
                )
            if max_tokens > self.max_tokens_limit:
                raise HTTPException(
                    status_code=422,
                    detail="'max_tokens' exceeds the configured limit",
                )
        else:
            params["max_tokens"] = 512

        temperature = params.get("temperature")
        if temperature is not None:
            if (
                not isinstance(temperature, (int, float))
                or isinstance(temperature, bool)
                or temperature < 0.0
                or temperature > 2.0
            ):
                raise HTTPException(
                    status_code=422,
                    detail="'temperature' must be a number between 0.0 and 2.0",
                )

        top_p = params.get("top_p")
        if top_p is not None:
            if (
                not isinstance(top_p, (int, float))
                or isinstance(top_p, bool)
                or top_p <= 0.0
                or top_p > 1.0
            ):
                raise HTTPException(
                    status_code=422,
                    detail="'top_p' must be a number in (0.0, 1.0]",
                )

        return params

    async def chat(self, payload: Any) -> JSONResponse | StreamingResponse:
        """Handle both non-streaming JSON completions and NDJSON streaming completions."""
        params = self._validate_payload(payload)
        is_stream = params["stream"]
        engine = self._get_engine()

        if not is_stream:
            async with self._generation_lock:
                result = await run_in_threadpool(
                    lambda: engine.llm.create_chat_completion(**params)
                )
            return JSONResponse(content=result)

        async def generate_response():
            try:
                async with self._generation_lock:
                    raw_gen = engine.stream_raw_chunks(params)
                    try:
                        async for chunk in iterate_in_threadpool(raw_gen):
                            yield (
                                json.dumps(
                                    chunk,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                    finally:
                        close = getattr(raw_gen, "close", None)
                        if callable(close):
                            try:
                                await run_in_threadpool(close)
                            except Exception:
                                pass
            except asyncio.CancelledError:
                logger.info("Streaming client disconnected; cancelling response")
                raise
            except Exception:
                logger.exception("Streaming generation failed")
                yield (
                    json.dumps(
                        {
                            "error": {
                                "code": "generation_failed",
                                "message": "The model could not complete this response.",
                            }
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        return StreamingResponse(
            generate_response(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
