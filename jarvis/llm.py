"""Local 4-bit GGUF LLM Engine (Qwen3-4B) with Streaming, Prefix KV-Cache & Performance Telemetry."""

from __future__ import annotations

import asyncio
import os
import re
import statistics
import threading
import time
from collections.abc import Iterator
from typing import Any, Dict, List, Optional, Tuple

import torch
from llama_cpp import Llama

try:
    from llama_cpp.llama_cache import LlamaRAMCache
except ImportError:
    from llama_cpp import LlamaCache as LlamaRAMCache  # type: ignore[no-redef]

from .config import LLMConfig
from .kv_memory import PersonalKVCacheManager


class LLMBackend:
    """Minimal backend interface for synchronous and streaming LLM generation."""

    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Return one complete assistant message."""
        raise NotImplementedError

    def stream(self, messages: List[Dict[str, str]], **kwargs: Any) -> Iterator[str]:
        """Yield incremental assistant text deltas."""
        raise NotImplementedError


class LocalLLMEngine(LLMBackend):
    """
    Manages Qwen3-4B-Instruct-2507 (Q4_K_M GGUF) via llama-cpp-python
    with GPU offloading, prefix KV-cache reuse, and incremental token streaming.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        model: Optional[Any] = None,
    ):
        self.config = config or LLMConfig()
        self.kv_manager = PersonalKVCacheManager(
            max_entries=self.config.max_kv_checkpoints
        )
        self.is_4bit_quantized: bool = True

        # Locks serialize access to the single underlying llama.cpp instance
        self.generation_lock = asyncio.Lock()
        self._sync_lock = threading.RLock()

        # Performance telemetry from the most recent streaming call
        self.last_stream_metrics: Dict[str, Any] = {}

        self.llm = model if model is not None else self._load_model()

    def _load_model(self) -> Llama:
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(
                f"GGUF model not found at: {self.config.model_path}"
            )

        llm = Llama(
            model_path=self.config.model_path,
            n_gpu_layers=self.config.n_gpu_layers,
            n_ctx=self.config.n_ctx,
            verbose=self.config.verbose,
        )

        # Enable llama.cpp's native in-RAM prefix KV-cache
        if self.config.use_prefix_kv_cache or self.config.use_personal_kv_cache:
            llm.set_cache(LlamaRAMCache(capacity_bytes=512 * 1024 * 1024))

        return llm

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_memory_footprint_mb(self) -> float:
        """Return the GGUF model footprint in megabytes."""
        if os.path.exists(self.config.model_path):
            return os.path.getsize(self.config.model_path) / (1024.0 * 1024.0)
        return 0.0

    def apply_template(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Render messages into ChatML string format used by Qwen3-Instruct."""
        parts: List[str] = [
            f"<|im_start|>{msg.get('role', 'user')}\n{msg.get('content', '')}<|im_end|>"
            for msg in messages
        ]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def encode_messages(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> List[int]:
        """Tokenize rendered ChatML messages using the embedded GGUF tokenizer."""
        prompt = self.apply_template(
            messages, add_generation_prompt=add_generation_prompt
        )
        return self.llm.tokenize(prompt.encode("utf-8"), add_bos=False)

    def _record_kv_prefix_telemetry(
        self, messages: List[Dict[str, str]], memory_ids: Tuple[int, ...]
    ) -> None:
        """Delegate prefix token matching and LRU bookkeeping to PersonalKVCacheManager."""
        full_tokens = self.encode_messages(messages, add_generation_prompt=True)
        sys_tokens: Optional[List[int]] = None
        if messages and messages[0].get("role") == "system":
            sys_tokens = self.encode_messages(
                [messages[0]], add_generation_prompt=False
            )
        self.kv_manager.record_prefix_usage(
            full_token_ids=full_tokens,
            sys_token_ids=sys_tokens,
            memory_ids=memory_ids,
        )

    def _resolve_sampling_params(
        self,
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        do_sample: bool,
    ) -> Tuple[int, float, float]:
        max_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else self.config.max_new_tokens
        )
        max_tokens = max(1, min(int(max_tokens), self.config.max_tokens_limit))
        temp = temperature if temperature is not None else self.config.temperature
        effective_temp = float(temp) if (do_sample and temp > 0.0) else 0.0
        effective_top_p = (
            float(self.config.top_p) if (do_sample and temp > 0.0) else 1.0
        )
        return max_tokens, effective_temp, effective_top_p

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: bool = True,
        memory_ids: Tuple[int, ...] = (),
        use_kv_cache: Optional[bool] = None,
    ) -> str:
        """Generate a complete non-streaming chat completion."""
        enable_kv = (
            (self.config.use_prefix_kv_cache or self.config.use_personal_kv_cache)
            if use_kv_cache is None
            else use_kv_cache
        )
        max_tokens, temp, top_p = self._resolve_sampling_params(
            max_new_tokens, temperature, do_sample
        )

        with self._sync_lock:
            if enable_kv:
                self._record_kv_prefix_telemetry(messages, memory_ids)

            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                stream=False,
            )

        content = response["choices"][0]["message"].get("content") or ""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        return content.strip()

    def stream_raw_chunks(self, params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Yield raw completion chunk dicts from `Llama.create_chat_completion(stream=True)`.
        Ensures the underlying C++/Python generator is closed on exit or disconnect.
        """
        call_params = dict(params)
        call_params["stream"] = True

        with self._sync_lock:
            iterator = self.llm.create_chat_completion(**call_params)
            try:
                for chunk in iterator:
                    yield chunk
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: bool = True,
        memory_ids: Tuple[int, ...] = (),
        use_kv_cache: Optional[bool] = None,
    ) -> Iterator[str]:
        """
        Stream incremental user-facing text deltas from the model while suppressing
        any `<think>...</think>` blocks and recording latency/throughput telemetry.
        """
        enable_kv = (
            (self.config.use_prefix_kv_cache or self.config.use_personal_kv_cache)
            if use_kv_cache is None
            else use_kv_cache
        )
        max_tokens, temp, top_p = self._resolve_sampling_params(
            max_new_tokens, temperature, do_sample
        )

        t_start = time.perf_counter()
        t_first_token: Optional[float] = None
        t_prev_token: Optional[float] = None
        inter_token_ms: List[float] = []
        chunk_count = 0

        in_think_block = False
        pending_buffer = ""

        with self._sync_lock:
            if enable_kv:
                self._record_kv_prefix_telemetry(messages, memory_ids)

            for chunk in self.stream_raw_chunks(
                {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temp,
                    "top_p": top_p,
                }
            ):
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                text = choices[0].get("delta", {}).get("content")
                if not text:
                    continue

                now = time.perf_counter()
                if t_first_token is None:
                    t_first_token = now
                elif t_prev_token is not None:
                    inter_token_ms.append((now - t_prev_token) * 1000.0)
                t_prev_token = now
                chunk_count += 1

                # Filter out accidental <think>...</think> blocks across chunk boundaries
                pending_buffer += text
                if "<think>" in pending_buffer:
                    in_think_block = True
                if in_think_block:
                    if "</think>" in pending_buffer:
                        _, after = pending_buffer.split("</think>", 1)
                        in_think_block = False
                        pending_buffer = ""
                        if after:
                            yield after
                    continue

                # Hold back partial prefix of "<think>" if buffer starts with "<"
                if pending_buffer.startswith("<") and "<think>".startswith(pending_buffer):
                    continue

                out_text = pending_buffer
                pending_buffer = ""
                if out_text:
                    yield out_text

            if pending_buffer and not in_think_block:
                yield pending_buffer

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0
        ttft_ms = (
            (t_first_token - t_start) * 1000.0
            if t_first_token is not None
            else total_ms
        )
        tps = chunk_count / max((t_end - t_start), 1e-6)
        median_itl = statistics.median(inter_token_ms) if inter_token_ms else 0.0
        p95_itl = (
            sorted(inter_token_ms)[int(0.95 * (len(inter_token_ms) - 1))]
            if inter_token_ms
            else 0.0
        )

        self.last_stream_metrics = {
            "ttft_ms": round(ttft_ms, 2),
            "total_generation_ms": round(total_ms, 2),
            "output_tokens": chunk_count,
            "tokens_per_sec": round(tps, 2),
            "inter_token_median_ms": round(median_itl, 2),
            "inter_token_p95_ms": round(p95_itl, 2),
            "footprint_mb": round(self.get_memory_footprint_mb(), 1),
        }

    # Standard LLMBackend interface aliases
    generate = generate_chat
    stream = stream_chat
