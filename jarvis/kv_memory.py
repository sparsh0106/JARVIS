"""External & Personal KV-Cache Retrieval and Prefix KV-Cache Telemetry Manager."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class KVCacheEntry:
    """Represents a cached token prefix and associated personal memory IDs."""

    key: str
    token_ids: Tuple[int, ...]
    memory_ids: Tuple[int, ...] = field(default_factory=tuple)
    hit_count: int = 0

    @property
    def seq_length(self) -> int:
        return len(self.token_ids)


class PersonalKVCacheManager:
    """
    Manages reusable KV-Cache prefix telemetry and bounded LRU entries for:
    1. Static System & Tool Schema Prefix (avoids re-prefilling on every agent round)
    2. Retrieved Personal Memory / Concept Evidence prefixes
    3. Multi-round agent conversation prefixes within a turn
    """

    def __init__(self, max_entries: int = 32):
        self.max_entries = max(4, max_entries)
        self._entries: OrderedDict[str, KVCacheEntry] = OrderedDict()
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.saved_prefill_tokens: int = 0
        self._turn_counter: int = 0

    @staticmethod
    def make_key(prefix_tag: str, memory_ids: Tuple[int, ...] = ()) -> str:
        raw = f"{prefix_tag}|mems={','.join(map(str, memory_ids))}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _find_longest_match(
        self, full_token_ids: Tuple[int, ...]
    ) -> Tuple[Optional[KVCacheEntry], int]:
        """Find the cached entry sharing the longest common token prefix (>= 32 tokens)."""
        max_usable_len = len(full_token_ids) - 1
        if max_usable_len <= 0:
            return None, 0

        best_entry: Optional[KVCacheEntry] = None
        best_match_len: int = 0

        for entry in self._entries.values():
            cached_tokens = entry.token_ids
            limit = min(len(cached_tokens), max_usable_len)
            if limit <= best_match_len:
                continue

            if (
                full_token_ids[: len(cached_tokens)] == cached_tokens
                and len(cached_tokens) <= max_usable_len
            ):
                best_entry = entry
                best_match_len = len(cached_tokens)
            else:
                common = 0
                for a, b in zip(full_token_ids[:limit], cached_tokens[:limit]):
                    if a != b:
                        break
                    common += 1
                if common > best_match_len and common >= 32:
                    best_entry = entry
                    best_match_len = common

        return best_entry, best_match_len

    def record_prefix_usage(
        self,
        full_token_ids: List[int] | Tuple[int, ...],
        sys_token_ids: Optional[List[int] | Tuple[int, ...]] = None,
        memory_ids: Tuple[int, ...] = (),
    ) -> int:
        """
        Record prefix KV-cache hit/miss telemetry for GGUF / llama.cpp inference
        and register the system + turn token prefixes in the bounded LRU table.
        Returns the number of matched prefix tokens saved.
        """
        tokens_tuple = tuple(full_token_ids)
        best_entry, best_match_len = self._find_longest_match(tokens_tuple)

        if best_entry is not None and best_match_len > 0:
            best_entry.hit_count += 1
            self._entries.move_to_end(best_entry.key)
            self.cache_hits += 1
            self.saved_prefill_tokens += best_match_len
        else:
            self.cache_misses += 1

        if sys_token_ids:
            sys_key = self.make_key("sys_mem", memory_ids)
            self._upsert_entry(sys_key, tuple(sys_token_ids), memory_ids)

        self._turn_counter += 1
        turn_key = self.make_key(f"turn_{self._turn_counter}", memory_ids)
        self._upsert_entry(turn_key, tokens_tuple, memory_ids)
        return best_match_len

    def _upsert_entry(
        self,
        key: str,
        token_ids: Tuple[int, ...],
        memory_ids: Tuple[int, ...],
    ) -> KVCacheEntry:
        existing = self._entries.get(key)
        hit_count = existing.hit_count if existing is not None else 0
        entry = KVCacheEntry(
            key=key,
            token_ids=token_ids,
            memory_ids=tuple(memory_ids),
            hit_count=hit_count,
        )
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return entry

    def stats(self) -> Dict[str, Any]:
        return {
            "entries": len(self._entries),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "saved_prefill_tokens": self.saved_prefill_tokens,
        }

    def clear(self) -> None:
        self._entries.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.saved_prefill_tokens = 0
        self._turn_counter = 0
