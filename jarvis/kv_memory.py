"""Personal KV-Cache & Prefix Telemetry Manager for Local GGUF Prompt Caching."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Sequence, Tuple


class PersonalKVCacheManager:
    """
    Tracks prefix reuse telemetry for llama.cpp's internal prompt cache.
    Monitors hits/misses when system prompts and retrieved personal memory IDs
    are reused across turns to avoid re-prefilling tokens.
    """

    def __init__(self, max_entries: int = 32):
        self.max_entries = max(4, max_entries)
        self._prefixes: OrderedDict[str, int] = OrderedDict()
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.saved_prefill_tokens: int = 0

    @staticmethod
    def _make_key(memory_ids: Sequence[int]) -> str:
        return ",".join(map(str, sorted(memory_ids))) if memory_ids else "base"

    def record_prefix_usage(
        self,
        memory_ids: Sequence[int] | Tuple[int, ...] = (),
        token_count: int = 0,
    ) -> None:
        """Record whether this memory/prompt prefix was seen before or is a new prefill."""
        key = self._make_key(memory_ids)
        tokens = max(0, int(token_count))

        if key in self._prefixes:
            self.cache_hits += 1
            saved = self._prefixes[key] or tokens
            self.saved_prefill_tokens += saved
            self._prefixes.move_to_end(key)
        else:
            self.cache_misses += 1
            self._prefixes[key] = tokens
            self._prefixes.move_to_end(key)
            if len(self._prefixes) > self.max_entries:
                self._prefixes.popitem(last=False)

    def stats(self) -> Dict[str, Any]:
        """Return cumulative telemetry matching CLI /kv and evaluation metrics."""
        return {
            "entries": len(self._prefixes),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "saved_prefill_tokens": self.saved_prefill_tokens,
        }

    def clear(self) -> None:
        """Reset all prefix cache counters and stored entries."""
        self._prefixes.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.saved_prefill_tokens = 0
