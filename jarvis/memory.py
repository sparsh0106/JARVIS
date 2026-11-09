"""Hybrid FAISS + TF-IDF + Cross-Encoder Memory System with Query-Aware Selection & MMR."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import MemoryConfig

VALID_MEMORY_TYPES: Set[str] = {"episodic", "semantic", "procedural", "preference", "goal"}


@dataclass
class QueryPlan:
    """Result of query-aware memory routing and intent analysis."""
    needs_memory: bool
    preferred_types: Optional[Set[str]]
    dense_weight: float
    tfidf_weight: float
    recency_boost: float
    min_relevance: float
    reason: str


class QueryAwareRouter:
    """Analyzes user queries to decide whether memory is needed and how to weight signals."""

    _PURE_UTILITY_PATTERNS = [
        re.compile(r"^\s*(?:what\s+time\s+is\s+it|current\s+time|tell\s+me\s+the\s+time)\b[0-9+\-*/().%\s\w?]*$", re.I),
        re.compile(r"^\s*(?:what\s+(?:is\s+)?(?:today'?s\s+)?date|what\s+day\s+is\s+it)\s*\??\s*$", re.I),
        re.compile(r"^\s*(?:calculate|compute|evaluate)\s+[0-9+\-*/().%\s]+\??\s*$", re.I),
        re.compile(r"^\s*[0-9]+\s*[+\-*/%]\s*[0-9+\-*/().%\s]+\s*\??\s*$", re.I),
        re.compile(r"^\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))\s*[!.]?\s*$", re.I),
    ]

    _TYPE_HINTS: Dict[str, Set[str]] = {
        "goal": {"goal", "goals", "plan", "planning", "future", "long-term", "longer-term", "aim", "roadmap", "objective", "eventually"},
        "preference": {"prefer", "like", "favorite", "interest", "interested", "hobby", "enjoy", "dislike", "style"},
        "procedural": {"how do i", "workflow", "setup", "command", "steps", "procedure", "habit", "routine"},
        "episodic": {"yesterday", "last time", "recently", "happened", "started", "history", "when did"},
        "semantic": {"my name", "who am i", "my age", "how old", "what am i building", "about me", "profession", "occupation", "engineer", "programmer"},
    }

    @classmethod
    def analyze(cls, query: str, config: MemoryConfig) -> QueryPlan:
        q = query.strip()
        if not q or any(pat.match(q) for pat in cls._PURE_UTILITY_PATTERNS):
            return QueryPlan(
                needs_memory=False, preferred_types=None, dense_weight=config.dense_weight,
                tfidf_weight=config.tfidf_weight, recency_boost=1.0, min_relevance=config.min_relevance_threshold,
                reason="empty_or_utility_query",
            )

        q_norm = re.sub(r"(what)(is)(my)(age|name)", r"\1 \2 \3 \4", q.lower())
        matched = {mtype for mtype, kws in cls._TYPE_HINTS.items() if any(kw in q_norm for kw in kws)}
        tokens = re.findall(r"\w+", q_norm)
        tfidf_w = min(0.50, config.tfidf_weight + 0.12) if len(tokens) <= 4 else config.tfidf_weight

        return QueryPlan(
            needs_memory=True,
            preferred_types=matched or None,
            dense_weight=1.0 - tfidf_w,
            tfidf_weight=tfidf_w,
            recency_boost=1.5 if "episodic" in matched else 1.0,
            min_relevance=config.min_relevance_threshold,
            reason=f"matched_types={sorted(matched)}" if matched else "general_query",
        )


class FAISSMemoryStore:
    """Persistent hybrid memory store with exact cross-modality scoring, Cross-Encoder reranking, and MMR."""

    _SINGLETON_FACT_PATTERNS = [
        ("user_age", re.compile(r"\b(?:age\s+is\s+\d+|\d+\s+years?\s+old)\b", re.I)),
        ("user_name", re.compile(r"\b(?:name\s+is\s+[a-z\s]+)\b", re.I)),
    ]

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        base_dir: Optional[str | Path] = None,
        embedder: Optional[SentenceTransformer] = None,
        reranker: Optional[CrossEncoder] = None,
        device: Optional[str] = None,
    ):
        self.config = config or MemoryConfig()
        if base_dir is not None:
            self.config.base_dir = Path(base_dir)

        self.base_dir = Path(self.config.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "index.faiss"
        self.metadata_path = self.base_dir / "metadata.json"

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = embedder or SentenceTransformer(self.config.embedding_model, device=self.device)
        self.embedder.eval()
        self.embedding_dim = int(self._encode_one("warmup").shape[1])

        self.reranker: Optional[CrossEncoder] = reranker
        if self.reranker is None and self.config.use_cross_encoder:
            try:
                self.reranker = CrossEncoder(self.config.reranker_model, device=self.device)
            except Exception:
                self.reranker = None

        self.index: faiss.IndexIDMap2 = faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedding_dim))
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Any = None
        self._tfidf_dirty: bool = True
        self._unflushed_reads: int = 0
        self.next_id: int = 1

        self._load()

    def _encode_one(self, text: str) -> np.ndarray:
        return self.embedder.encode(text, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32).reshape(1, -1)

    def _load(self) -> None:
        if self.index_path.exists() and self.metadata_path.exists():
            loaded_index = faiss.read_index(str(self.index_path))
            if loaded_index.d != self.embedding_dim:
                raise ValueError(f"FAISS dimension mismatch: index={loaded_index.d}, model={self.embedding_dim}.")
            self.index = loaded_index
            raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.metadata = {str(k): v for k, v in raw.items()}
            self.next_id = max(int(k) for k in self.metadata) + 1 if self.metadata else 1
            if self.index.ntotal != len(self.metadata):
                self._rebuild_index_from_metadata()
            else:
                self._rebuild_tfidf()
        elif self.metadata_path.exists():
            raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.metadata = {str(k): v for k, v in raw.items()}
            self._rebuild_index_from_metadata()
        else:
            self._tfidf_dirty = False

    def _rebuild_index_from_metadata(self) -> None:
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedding_dim))
        if not self.metadata:
            self.next_id = 1
            return
        ids = self._ordered_ids()
        texts = [self.metadata[str(mid)]["text"] for mid in ids]
        vectors = self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        self.index.add_with_ids(vectors, np.array(ids, dtype=np.int64))
        self.next_id = max(ids) + 1
        self._rebuild_tfidf()
        self.save(save_index=True)

    def _ordered_ids(self) -> List[int]:
        return sorted(int(k) for k in self.metadata)

    def _rebuild_tfidf(self) -> None:
        ids = self._ordered_ids()
        if not ids:
            self.vectorizer, self.tfidf_matrix, self._tfidf_dirty = None, None, False
            return
        texts = [self.metadata[str(mid)]["text"] for mid in ids]
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self._tfidf_dirty = False

    def _atomic_write(self, target: Path, prefix: str, writer_fn: Callable[[str], None]) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(self.base_dir), prefix=prefix, suffix=".tmp")
        os.close(fd)
        try:
            writer_fn(tmp)
            os.replace(tmp, str(target))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def save(self, save_index: bool = True) -> None:
        """Atomically persist index and metadata using write-to-temp-and-replace."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if save_index:
            self._atomic_write(self.index_path, ".index_", lambda tmp: faiss.write_index(self.index, tmp))
        self._atomic_write(
            self.metadata_path, ".meta_",
            lambda tmp: Path(tmp).write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        )
        self._unflushed_reads = 0

    def flush(self) -> None:
        """Flush buffered read access counts to metadata.json without rewriting FAISS."""
        if self._unflushed_reads > 0:
            self.save(save_index=False)

    @staticmethod
    def _recency(timestamp_iso: str, half_life_days: float = 30.0) -> float:
        try:
            ts = datetime.fromisoformat(timestamp_iso)
        except ValueError:
            return 0.5
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = max((datetime.now(timezone.utc) - ts).total_seconds() / 86400.0, 0.0)
        return math.exp(-math.log(2) * age_days / max(half_life_days, 1e-3))

    @staticmethod
    def _normalize_query_text(query: str) -> str:
        q = re.sub(r"(?i)^(what)(is)(my)(.+)$", r"\1 \2 \3 \4", query.strip())
        return re.sub(r"(?i)^(who)(am)(i)(.*)$", r"\1 \2 \3 \4", q).strip()

    def _find_singleton_conflict(self, text: str, memory_type: str) -> Optional[int]:
        if memory_type != "semantic" or len(text.split()) > 10:
            return None
        for _, pat in self._SINGLETON_FACT_PATTERNS:
            if pat.search(text):
                for mid_str, item in self.metadata.items():
                    if item.get("memory_type") == "semantic" and len(item["text"].split()) <= 10 and pat.search(item["text"]):
                        return int(mid_str)
        return None

    def add(
        self,
        text: str,
        memory_type: str = "semantic",
        importance: float = 0.5,
        source: str = "conversation",
        duplicate_threshold: Optional[float] = None,
        supersede_id: Optional[int] = None,
    ) -> int:
        """Add or supersede a memory with semantic deduplication and atomic persistence."""
        text = text.strip()
        if not text:
            raise ValueError("Memory text cannot be empty.")

        mtype = memory_type if memory_type in VALID_MEMORY_TYPES else "semantic"
        imp = float(np.clip(importance, 0.0, 1.0))
        dup_thresh = self.config.duplicate_threshold if duplicate_threshold is None else duplicate_threshold

        target_id = supersede_id or self._find_singleton_conflict(text, mtype)
        if target_id is not None and str(target_id) in self.metadata:
            self.update(memory_id=target_id, text=text, memory_type=mtype, importance=imp, source=source)
            return target_id

        vector = self._encode_one(text)
        now_iso = datetime.now(timezone.utc).isoformat()

        if self.index.ntotal > 0:
            distances, ids = self.index.search(vector, 1)
            nearest_score, nearest_id = float(distances[0][0]), int(ids[0][0])
            if nearest_id != -1 and nearest_score >= dup_thresh:
                existing = self.metadata[str(nearest_id)]
                existing["importance"] = max(float(existing["importance"]), imp)
                existing["access_count"] = int(existing.get("access_count", 0)) + 1
                existing["last_accessed_at"] = now_iso
                if len(text) > len(existing["text"]) and source != "seed":
                    existing["text"], existing["updated_at"] = text, now_iso
                    id_arr = np.array([nearest_id], dtype=np.int64)
                    self.index.remove_ids(id_arr)
                    self.index.add_with_ids(vector, id_arr)
                    self._tfidf_dirty = True
                    self.save(save_index=True)
                else:
                    self.save(save_index=False)
                return nearest_id

        memory_id = self.next_id
        self.next_id += 1
        self.index.add_with_ids(vector, np.array([memory_id], dtype=np.int64))
        self.metadata[str(memory_id)] = {
            "id": memory_id, "text": text, "memory_type": mtype, "created_at": now_iso,
            "updated_at": now_iso, "last_accessed_at": now_iso, "importance": imp,
            "access_count": 0, "source": source,
        }
        self._tfidf_dirty = True
        self.save(save_index=True)
        return memory_id

    def update(
        self,
        memory_id: int,
        text: Optional[str] = None,
        memory_type: Optional[str] = None,
        importance: Optional[float] = None,
        source: Optional[str] = None,
    ) -> bool:
        """Update an existing memory's text, vector, or metadata in-place."""
        key = str(memory_id)
        if key not in self.metadata:
            return False

        item = self.metadata[key]
        now_iso, index_changed = datetime.now(timezone.utc).isoformat(), False

        if text is not None and text.strip() and text.strip() != item["text"]:
            item["text"] = text.strip()
            vector = self._encode_one(item["text"])
            id_arr = np.array([memory_id], dtype=np.int64)
            self.index.remove_ids(id_arr)
            self.index.add_with_ids(vector, id_arr)
            self._tfidf_dirty = index_changed = True

        if memory_type is not None and memory_type in VALID_MEMORY_TYPES:
            item["memory_type"] = memory_type
        if importance is not None:
            item["importance"] = float(np.clip(importance, 0.0, 1.0))
        if source is not None:
            item["source"] = source

        item["updated_at"] = item["last_accessed_at"] = now_iso
        self.save(save_index=index_changed)
        return True

    def delete(self, memory_id: int) -> bool:
        """Remove a memory by ID from both FAISS and metadata."""
        removed = self.index.remove_ids(np.array([memory_id], dtype=np.int64))
        if removed:
            self.metadata.pop(str(memory_id), None)
            self._tfidf_dirty = True
            self.save(save_index=True)
        return bool(removed)

    def reset(self) -> None:
        """Clear all memories from both RAM and disk."""
        for path in (self.index_path, self.metadata_path):
            if path.exists():
                path.unlink()
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedding_dim))
        self.metadata = {}
        self.vectorizer, self.tfidf_matrix = None, None
        self._tfidf_dirty = False
        self._unflushed_reads = 0
        self.next_id = 1

    def _reconstruct_vector(self, memory_id: int) -> np.ndarray:
        return self.index.reconstruct(int(memory_id))

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        min_relevance: Optional[float] = None,
        use_reranker: Optional[bool] = None,
        use_mmr: bool = True,
        query_aware: bool = True,
        record_access: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute hybrid dense + sparse + Cross-Encoder retrieval with RRF fusion and MMR diversity."""
        if self.index.ntotal == 0 or not self.metadata:
            return []

        clean_query = self._normalize_query_text(query)
        if not clean_query:
            return []

        if self._tfidf_dirty or self.vectorizer is None:
            self._rebuild_tfidf()

        plan = QueryAwareRouter.analyze(clean_query, self.config) if query_aware else QueryPlan(
            needs_memory=True, preferred_types=None, dense_weight=self.config.dense_weight,
            tfidf_weight=self.config.tfidf_weight, recency_boost=1.0, min_relevance=self.config.min_relevance_threshold,
            reason="manual",
        )
        effective_min_rel = min_relevance if min_relevance is not None else plan.min_relevance

        ordered_ids = self._ordered_ids()
        eligible_ids = [mid for mid in ordered_ids if self.metadata[str(mid)].get("memory_type") == memory_type] if memory_type else ordered_ids
        if not eligible_ids:
            return []

        eligible_set = set(eligible_ids)
        query_vector = self._encode_one(clean_query)
        search_k = min(self.index.ntotal, max(self.config.candidate_k * 2, len(ordered_ids)))
        dense_raw, dense_ids = self.index.search(query_vector, search_k)
        dense_all = {int(mid): float(sc) for mid, sc in zip(dense_ids[0], dense_raw[0]) if int(mid) in eligible_set}
        dense_ranked = sorted(dense_all.items(), key=lambda x: x[1], reverse=True)[: self.config.candidate_k]
        dense_ranks = {mid: rank + 1 for rank, (mid, _) in enumerate(dense_ranked)}

        id_to_pos = {mid: idx for idx, mid in enumerate(ordered_ids)}
        raw_tfidf = cosine_similarity(self.vectorizer.transform([clean_query]), self.tfidf_matrix)[0] if (self.vectorizer and self.tfidf_matrix is not None) else None
        sparse_all = {mid: float(raw_tfidf[id_to_pos[mid]]) for mid in eligible_ids} if raw_tfidf is not None else {mid: 0.0 for mid in eligible_ids}
        sparse_ranked = sorted(sparse_all.items(), key=lambda x: x[1], reverse=True)[: self.config.candidate_k]
        sparse_ranks = {mid: rank + 1 for rank, (mid, sc) in enumerate(sparse_ranked) if sc > 0.0}

        candidate_ids = list({mid for mid, _ in dense_ranked} | {mid for mid, _ in sparse_ranked})
        if not candidate_ids:
            return []

        for mid in candidate_ids:
            if mid not in dense_all:
                dense_all[mid] = float(np.dot(query_vector[0], self._reconstruct_vector(mid)))

        max_sparse = max((sparse_all.get(mid, 0.0) for mid in candidate_ids), default=0.0)
        max_dense = max((max(0.0, dense_all.get(mid, 0.0)) for mid in candidate_ids), default=0.0)

        enable_ce = self.config.use_cross_encoder if use_reranker is None else use_reranker
        ce_probs: Dict[int, float] = {}
        if enable_ce and self.reranker is not None:
            pairs = [(clean_query, self.metadata[str(mid)]["text"]) for mid in candidate_ids]
            raw_logits = np.asarray(self.reranker.predict(pairs), dtype=np.float32)
            probs = 1.0 / (1.0 + np.exp(-np.clip(raw_logits, -20.0, 20.0)))
            ce_probs = {mid: float(p) for mid, p in zip(candidate_ids, probs)}

        scored_candidates: List[Dict[str, Any]] = []
        rrf_k, fallback_rank = self.config.rrf_k, len(eligible_ids) + 1

        for mid in candidate_ids:
            item = self.metadata[str(mid)]
            tfidf_sc, dense_sc = sparse_all.get(mid, 0.0), dense_all.get(mid, 0.0)
            dense_clamped = max(0.0, dense_sc)

            sparse_cal = (0.5 * tfidf_sc + 0.5 * (tfidf_sc / max_sparse)) if max_sparse > 1e-6 else 0.0
            dense_cal = (0.65 * dense_clamped + 0.35 * (dense_clamped / max_dense)) if max_dense > 1e-6 else 0.0
            r_dense, r_sparse = dense_ranks.get(mid, fallback_rank), sparse_ranks.get(mid, fallback_rank)
            rrf_sc = (rrf_k / (rrf_k + r_dense) + rrf_k / (rrf_k + r_sparse)) / 2.0

            hybrid_rel = 0.85 * (plan.tfidf_weight * sparse_cal + plan.dense_weight * dense_cal) + 0.15 * rrf_sc
            relevance = (
                (1.0 - self.config.cross_encoder_weight) * hybrid_rel + self.config.cross_encoder_weight * ce_probs[mid]
                if mid in ce_probs else hybrid_rel
            )

            if plan.preferred_types and item.get("memory_type") in plan.preferred_types and relevance >= effective_min_rel * 0.85:
                relevance = min(1.0, relevance * 1.08)

            # Combined threshold gating
            if (
                (dense_clamped < 0.25 and tfidf_sc < 0.10 and ce_probs.get(mid, 0.0) < 0.15)
                or (mid in ce_probs and ce_probs[mid] < 0.03 and dense_clamped < 0.35 and tfidf_sc < 0.15)
                or (relevance < effective_min_rel and dense_clamped < 0.28 and tfidf_sc < 0.12)
            ):
                continue

            last_touch = item.get("last_accessed_at") or item.get("updated_at") or item.get("created_at")
            recency = self._recency(last_touch, half_life_days=self.config.recency_half_life_days)
            access_sc = min(1.0, math.log1p(int(item.get("access_count", 0))) / math.log1p(50.0))
            meta_prior = (
                self.config.beta_importance * float(item.get("importance", 0.5))
                + self.config.gamma_recency * min(1.0, recency * plan.recency_boost)
                + self.config.delta_access * access_sc
            )
            final_sc = self.config.alpha_relevance * relevance + min(1.0, relevance / max(effective_min_rel, 0.15)) * meta_prior

            scored_candidates.append({
                **item, "last_accessed_at": last_touch, "tfidf_score": tfidf_sc, "dense_score": dense_sc,
                "dense_score_normalized": dense_clamped, "reranker_score": ce_probs.get(mid),
                "relevance": relevance, "recency": recency, "score": final_sc,
            })

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        results = (
            self._mmr_select(scored_candidates, top_k=top_k)
            if (use_mmr and len(scored_candidates) > 1 and top_k > 1)
            else scored_candidates[:top_k]
        )

        if record_access and results:
            now_iso = datetime.now(timezone.utc).isoformat()
            for r in results:
                meta = self.metadata[str(r["id"])]
                meta["access_count"] = int(meta.get("access_count", 0)) + 1
                meta["last_accessed_at"] = now_iso
            self._unflushed_reads += 1
            if self._unflushed_reads >= self.config.flush_every_n_reads:
                self.save(save_index=False)

        return results

    def _mmr_select(self, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Diversify top_k results using Maximal Marginal Relevance over reconstructed FAISS vectors."""
        lam = self.config.mmr_lambda
        vecs = {c["id"]: self._reconstruct_vector(c["id"]) for c in candidates}
        selected: List[Dict[str, Any]] = [candidates[0]]
        remaining = candidates[1:]

        while remaining and len(selected) < top_k:
            best_idx = max(
                range(len(remaining)),
                key=lambda i: lam * remaining[i]["score"]
                - (1.0 - lam) * max(0.0, max(float(np.dot(vecs[remaining[i]["id"]], vecs[s["id"]])) for s in selected)),
            )
            selected.append(remaining.pop(best_idx))

        return selected

    search = hybrid_search

    def list_all(self) -> List[Dict[str, Any]]:
        return [self.metadata[str(mid)] for mid in self._ordered_ids()]

    @property
    def size(self) -> int:
        return int(self.index.ntotal)
