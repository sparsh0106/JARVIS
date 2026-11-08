"""Local 4-bit GGUF Model Management (Qwen3-4B-Instruct) with Token Streaming & KV-Cache."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .config import LLMConfig
from .kv_memory import PersonalKVCacheManager

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


class LLMBackend:
    """Wrapper managing llama-cpp-python runtime, GPU offloading, and context windows."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.llm = self._load_model()

    def _load_model(self) -> Any:
        if Llama is None:
            raise ImportError(
                "llama-cpp-python is required for local GGUF execution. "
                "Install it with: pip install llama-cpp-python"
            )

        model_path = os.path.expanduser(self.config.model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GGUF model file not found at: {model_path}. "
                "Please verify the path in your configuration or JARVIS_GGUF_PATH env var."
            )

        llm = Llama(
            model_path=model_path,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=self.config.verbose,
        )
        return llm


class LocalLLMEngine:
    """High-level GGUF engine providing structured text generation, token streaming, and KV-cache tracking."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        backend: Optional[LLMBackend] = None,
    ):
        self.config = config or LLMConfig()
        self.backend = backend or LLMBackend(config=self.config)
        self.lock = threading.Lock()
        self.kv_manager = PersonalKVCacheManager()

        # Telemetry metrics from the most recent stream turn
        self.last_stream_metrics: Dict[str, Any] = {
            "ttft_ms": 0.0,
            "total_generation_ms": 0.0,
            "output_tokens": 0,
            "tokens_per_sec": 0.0,
            "inter_token_median_ms": 0.0,
            "inter_token_p95_ms": 0.0,
            "footprint_mb": self.get_memory_footprint_mb(),
        }

    def get_memory_footprint_mb(self) -> float:
        """Estimate model memory footprint in MB."""
        try:
            model_path = os.path.expanduser(self.config.model_path)
            if os.path.exists(model_path):
                return os.path.getsize(model_path) / (1024 * 1024)
        except Exception:
            pass
        return 2380.0

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: bool = True,
        memory_ids: Optional[Tuple[int, ...]] = None,
    ) -> str:
        """Non-streaming generation returning the complete response string."""
        max_tokens = max_new_tokens or self.config.max_new_tokens
        temp = temperature if temperature is not None else self.config.temperature
        if not do_sample:
            temp = 0.0

        with self.lock:
            resp = self.backend.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=self.config.top_p,
                stream=False,
            )

        if memory_ids:
            self.kv_manager.record_prefix_usage(
                memory_ids=memory_ids,
                token_count=resp.get("usage", {}).get("prompt_tokens", 0),
            )

        return resp["choices"][0]["message"]["content"] or ""

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: bool = True,
        memory_ids: Optional[Tuple[int, ...]] = None,
    ) -> Iterator[str]:
        """Incremental token streaming generator yielding each text delta as soon as generated."""
        max_tokens = max_new_tokens or self.config.max_new_tokens
        temp = temperature if temperature is not None else self.config.temperature
        if not do_sample:
            temp = 0.0

        t_start = time.perf_counter()
        t_first_token: Optional[float] = None
        inter_token_times: List[float] = []
        token_count = 0
        last_token_time = t_start

        with self.lock:
            stream_gen = self.backend.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=self.config.top_p,
                stream=True,
            )

            for chunk in stream_gen:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    now = time.perf_counter()
                    if t_first_token is None:
                        t_first_token = now
                    else:
                        inter_token_times.append((now - last_token_time) * 1000.0)
                    last_token_time = now
                    token_count += 1
                    yield content

        t_end = time.perf_counter()
        gen_duration = t_end - t_start
        ttft_ms = (t_first_token - t_start) * 1000.0 if t_first_token else 0.0

        if inter_token_times:
            sorted_times = sorted(inter_token_times)
            p50 = sorted_times[len(sorted_times) // 2]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
        else:
            p50 = 0.0
            p95 = 0.0

        tok_per_sec = (token_count / gen_duration) if gen_duration > 0 else 0.0

        self.last_stream_metrics = {
            "ttft_ms": round(ttft_ms, 2),
            "total_generation_ms": round(gen_duration * 1000.0, 2),
            "output_tokens": token_count,
            "tokens_per_sec": round(tok_per_sec, 2),
            "inter_token_median_ms": round(p50, 2),
            "inter_token_p95_ms": round(p95, 2),
            "footprint_mb": self.get_memory_footprint_mb(),
        }

        if memory_ids:
            self.kv_manager.record_prefix_usage(
                memory_ids=memory_ids,
                token_count=token_count,
            )
