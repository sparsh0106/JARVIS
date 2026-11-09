"""Local GGUF Model Management (Qwen3-4B-Instruct) with Real-Time Token Streaming."""

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
    """Backward-compatible wrapper for llama-cpp-python runtime."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.llm = LocalLLMEngine._load_llama(self.config)


class LocalLLMEngine:
    """High-level GGUF engine providing structured text generation and word-by-word token streaming."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        backend: Optional[LLMBackend] = None,
    ):
        self.config = config or LLMConfig()
        self.llm = backend.llm if backend else self._load_llama(self.config)
        self.lock = threading.Lock()
        self.kv_manager = PersonalKVCacheManager()
        self.last_stream_metrics: Dict[str, Any] = {}

    @staticmethod
    def _load_llama(config: LLMConfig) -> Any:
        if Llama is None:
            raise ImportError("llama-cpp-python is required: pip install llama-cpp-python")
        path = os.path.expanduser(config.model_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"GGUF model file not found at: {path}")
        return Llama(
            model_path=path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            verbose=config.verbose,
        )

    def get_memory_footprint_mb(self) -> float:
        """Estimate model memory footprint in MB."""
        try:
            path = os.path.expanduser(self.config.model_path)
            if os.path.exists(path):
                return round(os.path.getsize(path) / (1024 * 1024), 1)
        except Exception:
            pass
        return 2380.0

    def _params(
        self, max_tokens: Optional[int], temp: Optional[float], sample: bool
    ) -> Tuple[int, float]:
        return (
            max_tokens or self.config.max_new_tokens,
            0.0 if not sample else (temp if temp is not None else self.config.temperature),
        )

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: bool = True,
        memory_ids: Optional[Tuple[int, ...]] = None,
    ) -> str:
        """Non-streaming generation returning the full response text."""
        tokens, temp = self._params(max_new_tokens, temperature, do_sample)
        with self.lock:
            resp = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=tokens,
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
        """Real-time token streaming yielding each word/character delta as generated."""
        tokens, temp = self._params(max_new_tokens, temperature, do_sample)
        t_start = time.perf_counter()
        t_first: Optional[float] = None
        count = 0

        with self.lock:
            stream = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=tokens,
                temperature=temp,
                top_p=self.config.top_p,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    if t_first is None:
                        t_first = time.perf_counter()
                    count += 1
                    yield delta

        duration = max(time.perf_counter() - t_start, 1e-4)
        ttft_ms = (t_first - t_start) * 1000.0 if t_first else 0.0

        self.last_stream_metrics = {
            "ttft_ms": round(ttft_ms, 2),
            "total_generation_ms": round(duration * 1000.0, 2),
            "output_tokens": count,
            "tokens_per_sec": round(count / duration, 2),
            "footprint_mb": self.get_memory_footprint_mb(),
        }

        if memory_ids:
            self.kv_manager.record_prefix_usage(memory_ids=memory_ids, token_count=count)
