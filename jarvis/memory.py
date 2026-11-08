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
from typing import Any, Dict, List, Optional, Set, Tuple

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

    _TYPE_HINTS: Dict[str, List[str]] = {
        "goal": ["goal", "goals", "plan", "planning", "future", "long-term", "longer-term", "aim", "roadmap", "objective", "eventually"],
        "preference": ["prefer", "like", "favorite", "interest", "interested", "hobby", "enjoy", "dislike", "style"],
        "procedural": ["how do i", "workflow", "setup", "command", "steps", "procedure", "habit", "routine"],
        "episodic": ["yesterday", "last time", "recently", "happened", "started", "history", "when did"],
        "semantic": ["my name", "who am i", "my age", "how old", "what am i building", "about me", "profession", "occupation", "engineer", "programmer"],
    }

    @classmethod
    def analyze(cls, query: str, config: MemoryConfig) -> QueryPlan:
        q = query.strip()
        if not q:
            return QueryPlan(
                needs_memory=False,
                preferred_types=None,
                dense_weight=config.dense_weight,
                tfidf_weight=config.tfidf_weight,
                recency_boost=1.0,
                min_relevance=config.min_relevance_threshold,
                reason="empty_query",
            )

        for pat in cls._PURE_UTILITY_PATTERNS:
            if pat.match(q):
                return QueryPlan(
                    needs_memory=False,
                    preferred_types=None,
                    dense_weight=config.dense_weight,
                    tfidf_weight=config.tfidf_weight,
                    recency_boost=1.0,
                    min_relevance=config.min_relevance_threshold,
                    reason="utility_or_greeting_query",
                )

        q_lower = q.lower()
        # Normalize spaceless queries like "whatismyage?"
        q_normalized = re.sub(r"(what)(is)(my)(age|name)", r"\1 \2 \3 \4", q_lower)

        matched_types: Set[str] = set()
        for mtype, keywords in cls._TYPE_HINTS.items():
            if any(kw in q_normalized for kw in keywords):
                matched_types.add(mtype)

        # Short or entity-heavy queries benefit from stronger lexical TF-IDF weight
        tokens = re.findall(r"\w+", q_normalized)
        if len(tokens) <= 4:
            tfidf_w = min(0.50, config.tfidf_weight + 0.12)
            dense_w = 1.0 - tfidf_w
        else:
            tfidf_w = config.tfidf_weight
            dense_w = config.dense_weight

        recency_boost = 1.5 if "episodic" in matched_types else 1.0

        return QueryPlan(needs_memory=True, preferred_types=matched_types if matched_types else None,
            dense_weight=dense_w, tfidf_weight=tfidf_w, recency_boost=recency_boost,
            min_relevance=config.min_relevance_threshold,
            reason=f"matched_types={sorted(matched_types)}" if matched_types else "general_query")


class FAISSMemoryStore:
    """Persistent hybrid memory store with exact cross-modality scoring, Cross-Encoder reranking, and MMR."""

    # Patterns for single-valued personal attributes that should supersede old entries
    _SINGLETON_FACT_PATTERNS = [("user_age", re.compile(r"\b(?:age\s+is\s+\d+|\d+\s+years?\s+old)\b", re.I)),
        ("user_name", re.compile(r"\b(?:name\s+is\s+[a-z\s]+)\b", re.I))]

    def __init__(
        self, config: Optional[MemoryConfig] = None, base_dir: Optional[str | Path] = None,
        embedder: Optional[SentenceTransformer] = None, reranker: Optional[CrossEncoder] = None,
        device: Optional[str] = None):
        self.config = config or MemoryConfig()
        if base_dir is not None:
            self.config.base_dir = Path(base_dir)

        self.base_dir = Path(self.config.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.base_dir / "index.faiss"
        self.metadata_path = self.base_dir / "metadata.json"

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = embedder or SentenceTransformer(
            self.config.embedding_model, device=self.device
        )
        self.embedder.eval()
        test_vec = self.embedder.encode(
            "warmup", normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        self.embedding_dim = int(test_vec.shape[0])

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

    def _load(self) -> None:
        if self.index_path.exists() and self.metadata_path.exists():
            loaded_index = faiss.read_index(str(self.index_path))
            if loaded_index.d != self.embedding_dim:
                raise ValueError(f"FAISS dimension mismatch: index={loaded_index.d}, embedding_model={self.embedding_dim}.")
            self.index = loaded_index
            raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.metadata = {str(k): v for k, v in raw.items()}
            if self.metadata:
                self.next_id = max(int(k) for k in self.metadata) + 1
            self._validate_consistency()
            self._rebuild_tfidf()
        elif self.metadata_path.exists() and not self.index_path.exists():
            # Self-heal if index file was missing but metadata survived
            raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.metadata = {str(k): v for k, v in raw.items()}
            self._rebuild_index_from_metadata()
        else:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedding_dim))
            self.metadata = {}
            self.next_id = 1
            self._tfidf_dirty = False

    def _rebuild_index_from_metadata(self) -> None:
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedding_dim))
        if not self.metadata:
            self.next_id = 1
            return
        ids = self._ordered_ids()
        texts = [self.metadata[str(mid)]["text"] for mid in ids]
        vectors = self.embedder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        self.index.add_with_ids(vectors, np.array(ids, dtype=np.int64))
        self.next_id = max(ids) + 1
        self._rebuild_tfidf()
        self.save(save_index=True)

    def _validate_consistency(self) -> None:
        if self.index.ntotal != len(self.metadata):
            # Automatically rebuild FAISS index from authoritative metadata.json
            self._rebuild_index_from_metadata()

    def _ordered_ids(self) -> List[int]:
        return sorted(int(k) for k in self.metadata)

    def _rebuild_tfidf(self) -> None:
        ids = self._ordered_ids()
        if not ids:
            self.vectorizer = None
            self.tfidf_matrix = None
            self._tfidf_dirty = False
            return

        texts = [self.metadata[str(mid)]["text"] for mid in ids]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self._tfidf_dirty = False

    def save(self, save_index: bool = True) -> None:
        """Atomically persist index and metadata using write-to-temp-and-replace."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if save_index:
            fd, tmp_idx = tempfile.mkstemp(dir=str(self.base_dir), prefix=".index_", suffix=".tmp")
            os.close(fd)
            try:
                faiss.write_index(self.index, tmp_idx)
                os.replace(tmp_idx, str(self.index_path))
            finally:
                if os.path.exists(tmp_idx):
                    os.unlink(tmp_idx)

        fd, tmp_meta = tempfile.mkstemp(dir=str(self.base_dir), prefix=".meta_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            os.replace(tmp_meta, str(self.metadata_path))
        finally:
            if os.path.exists(tmp_meta):
                os.unlink(tmp_meta)

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
        now = datetime.now(timezone.utc)
        age_days = max((now - ts).total_seconds() / 86400.0, 0.0)
        return math.exp(-math.log(2) * age_days / max(half_life_days, 1e-3))

    @staticmethod
    def _normalize_query_text(query: str) -> str:
        """Clean up spaceless queries like 'whatismyage?' so tokenizers/TF-IDF work reliably."""
        q = query.strip()
        q = re.sub(r"(?i)^(what)(is)(my)(.+)$", r"\1 \2 \3 \4", q)
        q = re.sub(r"(?i)^(who)(am)(i)(.*)$", r"\1 \2 \3 \4", q)
        return q.strip()

    def _find_singleton_conflict(self, text: str, memory_type: str) -> Optional[int]:
        """Detect if the new memory updates a single-valued personal fact (like age)."""
        if memory_type != "semantic":
            return None
        for _, pattern in self._SINGLETON_FACT_PATTERNS:
            if pattern.search(text):
                for mid_str, item in self.metadata.items():
                    if item.get("memory_type") == "semantic" and pattern.search(item["text"]):
                        # Only treat as singleton replacement if both texts are purely that fact
                        if len(item["text"].split()) <= 10 and len(text.split()) <= 10:
                            return int(mid_str)
        return None

    def add(self, text: str, memory_type: str = "semantic", importance: float = 0.5,
        source: str = "conversation", duplicate_threshold: Optional[float] = None, supersede_id: Optional[int] = None) -> int:
        """Add or supersede a memory with semantic deduplication and atomic persistence."""
        text = text.strip()
        if not text:
            raise ValueError("Memory text cannot be empty.")

        if memory_type not in VALID_MEMORY_TYPES:
            memory_type = "semantic"

        importance = float(np.clip(importance, 0.0, 1.0))
        dup_thresh = (duplicate_threshold if duplicate_threshold is not None else self.config.duplicate_threshold)

        # If explicitly superseding or matching a singleton personal fact, update in-place
        if supersede_id is None:
            supersede_id = self._find_singleton_conflict(text, memory_type)

        if supersede_id is not None and str(supersede_id) in self.metadata:
            self.update(memory_id=supersede_id, text=text, memory_type=memory_type, importance=importance, source=source)
            return supersede_id

        vector = (self.embedder.encode(text, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32).reshape(1, -1))
        now_iso = datetime.now(timezone.utc).isoformat()

        if self.index.ntotal > 0:
            distances, ids = self.index.search(vector, 1)
            nearest_score = float(distances[0][0])
            nearest_id = int(ids[0][0])
            if nearest_id != -1 and nearest_score >= dup_thresh:
                existing = self.metadata[str(nearest_id)]
                existing["importance"] = max(float(existing["importance"]), importance)
                existing["access_count"] = int(existing.get("access_count", 0)) + 1
                existing["last_accessed_at"] = now_iso
                # If the new wording is more informative and from user/tool, refresh the text & vector
                if len(text) > len(existing["text"]) and source != "seed":
                    existing["text"] = text
                    existing["updated_at"] = now_iso
                    self.index.remove_ids(np.array([nearest_id], dtype=np.int64))
                    self.index.add_with_ids(vector, np.array([nearest_id], dtype=np.int64))
                    self._tfidf_dirty = True
                    self.save(save_index=True)
                else:
                    self.save(save_index=False)
                return nearest_id

        memory_id = self.next_id
        self.next_id += 1

        self.index.add_with_ids(vector, np.array([memory_id], dtype=np.int64))
        self.metadata[str(memory_id)] = {
            "id": memory_id,
            "text": text,
            "memory_type": memory_type,
            "created_at": now_iso,
            "updated_at": now_iso,
            "last_accessed_at": now_iso,
            "importance": importance,
            "access_count": 0,
            "source": source,
        }
        self._tfidf_dirty = True
        self.save(save_index=True)
        return memory_id

    def update(self, memory_id: int, text: Optional[str] = None, memory_type: Optional[str] = None,
        importance: Optional[float] = None, source: Optional[str] = None) -> bool:
        """Update an existing memory's text, vector, or metadata in-place."""
        key = str(memory_id)
        if key not in self.metadata:
            return False

        item = self.metadata[key]
        now_iso = datetime.now(timezone.utc).isoformat()
        index_changed = False

        if text is not None and text.strip() and text.strip() != item["text"]:
            item["text"] = text.strip()
            vector = (self.embedder.encode(item["text"], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32).reshape(1, -1))
            self.index.remove_ids(np.array([memory_id], dtype=np.int64))
            self.index.add_with_ids(vector, np.array([memory_id], dtype=np.int64))
            self._tfidf_dirty = True
            index_changed = True

        if memory_type is not None and memory_type in VALID_MEMORY_TYPES:
            item["memory_type"] = memory_type
        if importance is not None:
            item["importance"] = float(np.clip(importance, 0.0, 1.0))
        if source is not None:
            item["source"] = source

        item["updated_at"] = now_iso
        item["last_accessed_at"] = now_iso
        self.metadata[key] = item
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
        for path in [self.index_path, self.metadata_path]:
            if path.exists():
                path.unlink()
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedding_dim))
        self.metadata = {}
        self.vectorizer = None
        self.tfidf_matrix = None
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
        """
        Execute hybrid retrieval with:
        1. Pre-filtering by memory_type (so candidate_k never starves filtered queries)
        2. Exact dense & sparse score computation for all candidates (no missing-modality 0 cliff)
        3. Calibrated RRF + score fusion + optional Cross-Encoder reranking
        4. Relevance threshold gating before applying importance/recency/access priors
        5. Maximal Marginal Relevance (MMR) diversification
        """
        if self.index.ntotal == 0 or not self.metadata:
            return []

        clean_query = self._normalize_query_text(query)
        if not clean_query:
            return []

        if self._tfidf_dirty or self.vectorizer is None:
            self._rebuild_tfidf()

        plan = (
            QueryAwareRouter.analyze(clean_query, self.config)
            if query_aware
            else QueryPlan(
                needs_memory=True,
                preferred_types=None,
                dense_weight=self.config.dense_weight,
                tfidf_weight=self.config.tfidf_weight,
                recency_boost=1.0,
                min_relevance=self.config.min_relevance_threshold,
                reason="manual",
            )
        )

        effective_min_rel = (
            min_relevance if min_relevance is not None else plan.min_relevance
        )

        # 1. Pre-filter valid IDs by memory_type BEFORE candidate truncation
        ordered_ids = self._ordered_ids()
        if memory_type is not None:
            eligible_ids = [mid for mid in ordered_ids if self.metadata[str(mid)].get("memory_type") == memory_type]
        else:
            eligible_ids = ordered_ids
        if not eligible_ids:
            return []

        eligible_set = set(eligible_ids)

        # 2. Dense retrieval (FAISS) + exact reconstruction for any eligible candidate
        query_vector = (self.embedder.encode(clean_query, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32).reshape(1, -1))
        search_k = min(self.index.ntotal, max(self.config.candidate_k * 2, len(ordered_ids)),)
        dense_raw, dense_ids = self.index.search(query_vector, search_k)
        dense_all: Dict[int, float] = {int(mid): float(score) for mid, score in zip(dense_ids[0], dense_raw[0]) if int(mid) in eligible_set}

        # Sort eligible dense candidates
        dense_ranked = sorted(dense_all.items(), key=lambda x: x[1], reverse=True)[: self.config.candidate_k]
        dense_scores = dict(dense_ranked)
        dense_ranks = {mid: rank + 1 for rank, (mid, _) in enumerate(dense_ranked)}

        # 3. Sparse retrieval (TF-IDF) across all documents, filtered to eligible_set
        id_to_pos = {mid: idx for idx, mid in enumerate(ordered_ids)}
        if self.vectorizer is not None and self.tfidf_matrix is not None:
            query_tfidf = self.vectorizer.transform([clean_query])
            raw_tfidf = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
            sparse_all: Dict[int, float] = {mid: float(raw_tfidf[id_to_pos[mid]]) for mid in eligible_ids}
        else:
            sparse_all = {mid: 0.0 for mid in eligible_ids}

        sparse_ranked = sorted(sparse_all.items(), key=lambda x: x[1], reverse=True)[: self.config.candidate_k]
        sparse_scores = dict(sparse_ranked)
        sparse_ranks = {mid: rank + 1 for rank, (mid, score) in enumerate(sparse_ranked) if score > 0.0}

        # 4. Union of top candidates from both modalities
        candidate_ids = list(set(dense_scores.keys()) | set(sparse_scores.keys()))
        if not candidate_ids:
            return []

        # Backfill exact dense cosine via IndexIDMap2.reconstruct(id) & exact sparse score
        for mid in candidate_ids:
            if mid not in dense_all:
                vec = self._reconstruct_vector(mid)
                dense_all[mid] = float(np.dot(query_vector[0], vec))

        # Compute min-max scaling over the candidate pool so TF-IDF and Dense are on commensurate [0, 1] scales
        max_sparse = max((sparse_all.get(mid, 0.0) for mid in candidate_ids), default=0.0)
        max_dense = max((max(0.0, dense_all.get(mid, 0.0)) for mid in candidate_ids), default=0.0)

        # 5. Optional Cross-Encoder Re-ranking on the candidate pool
        enable_ce = (self.config.use_cross_encoder if use_reranker is None else use_reranker)
        ce_probs: Dict[int, float] = {}
        if enable_ce and self.reranker is not None and candidate_ids:
            pairs = [(clean_query, self.metadata[str(mid)]["text"]) for mid in candidate_ids]
            raw_logits = np.asarray(self.reranker.predict(pairs), dtype=np.float32)
            
            # Sigmoid calibration maps cross-encoder logits to [0, 1] probability
            probs = 1.0 / (1.0 + np.exp(-np.clip(raw_logits, -20.0, 20.0)))
            ce_probs = {mid: float(prob) for mid, prob in zip(candidate_ids, probs)}

        scored_candidates: List[Dict[str, Any]] = []
        rrf_k = self.config.rrf_k

        for mid in candidate_ids:
            item = self.metadata[str(mid)]
            tfidf_score = sparse_all.get(mid, 0.0)
            dense_score = dense_all.get(mid, 0.0)

            # Positive cosine floor at 0.0 (orthogonal = 0.0, not 0.5!)
            dense_clamped = max(0.0, dense_score)

            # Relative calibration within candidate pool + absolute anchor
            sparse_calibrated = (0.5 * tfidf_score + 0.5 * (tfidf_score / max_sparse) if max_sparse > 1e-6 else 0.0)
            dense_calibrated = (0.65 * dense_clamped + 0.35 * (dense_clamped / max_dense) if max_dense > 1e-6 else 0.0)

            # RRF bonus rewards agreement between dense and lexical rankers
            r_dense = dense_ranks.get(mid, len(eligible_ids) + 1)
            r_sparse = sparse_ranks.get(mid, len(eligible_ids) + 1)
            rrf_score = (rrf_k / (rrf_k + r_dense) + rrf_k / (rrf_k + r_sparse)) / 2.0

            bi_relevance = (plan.tfidf_weight * sparse_calibrated + plan.dense_weight * dense_calibrated)

            # Blend bi-encoder hybrid relevance with RRF agreement
            hybrid_relevance = 0.85 * bi_relevance + 0.15 * rrf_score

            if mid in ce_probs:
                ce_w = self.config.cross_encoder_weight
                relevance = (1.0 - ce_w) * hybrid_relevance + ce_w * ce_probs[mid]
            else:
                relevance = hybrid_relevance

            # Slight query-aware type affinity boost if query hinted at this memory_type
            if (plan.preferred_types and item.get("memory_type") in plan.preferred_types and relevance >= effective_min_rel * 0.85):
                relevance = min(1.0, relevance * 1.08)

            # Gate out irrelevant memories BEFORE relative rank/metadata priors can boost them
            if dense_clamped < 0.25 and tfidf_score < 0.10 and ce_probs.get(mid, 0.0) < 0.15: continue
            if mid in ce_probs and ce_probs[mid] < 0.03 and dense_clamped < 0.35 and tfidf_score < 0.15: continue
            if relevance < effective_min_rel and dense_clamped < 0.28 and tfidf_score < 0.12: continue

            importance = float(item.get("importance", 0.5))
            last_touch = (item.get("last_accessed_at") or item.get("updated_at") or item.get("created_at"))
            recency = self._recency(last_touch, half_life_days=self.config.recency_half_life_days)
            access_count = int(item.get("access_count", 0))
            access_score = min(1.0, math.log1p(access_count) / math.log1p(50.0))

            # Metadata prior is scaled by relevance so a high-importance memory only wins ties among relevant hits
            metadata_prior = (self.config.beta_importance * importance + self.config.gamma_recency * min(1.0, recency * plan.recency_boost) + self.config.delta_access * access_score)
            prior_gate = min(1.0, relevance / max(effective_min_rel, 0.15))
            final_score = (self.config.alpha_relevance * relevance + prior_gate * metadata_prior)

            scored_candidates.append(
                {
                    "id": mid,
                    "text": item["text"],
                    "memory_type": item["memory_type"],
                    "created_at": item["created_at"],
                    "updated_at": item.get("updated_at", item["created_at"]),
                    "last_accessed_at": last_touch,
                    "importance": importance,
                    "access_count": access_count,
                    "source": item.get("source", "conversation"),
                    "tfidf_score": tfidf_score,
                    "dense_score": dense_score,
                    "dense_score_normalized": dense_clamped,
                    "reranker_score": ce_probs.get(mid),
                    "relevance": relevance,
                    "recency": recency,
                    "score": final_score,
                }
            )

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        # 6. Maximal Marginal Relevance (MMR) selection to prevent near-duplicate flooding
        if use_mmr and len(scored_candidates) > 1 and top_k > 1:
            results = self._mmr_select(scored_candidates, top_k=top_k)
        else:
            results = scored_candidates[:top_k]

        # 7. Update access counts in memory (buffered disk write)
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
            best_idx = 0
            best_mmr = -1e9
            for idx, cand in enumerate(remaining):
                c_vec = vecs[cand["id"]]
                max_sim = max(float(np.dot(c_vec, vecs[sel["id"]])) for sel in selected)
                mmr_val = lam * cand["score"] - (1.0 - lam) * max(0.0, max_sim)
                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_idx = idx
            selected.append(remaining.pop(best_idx))

        return selected

    search = hybrid_search

    def list_all(self) -> List[Dict[str, Any]]:
        return [self.metadata[str(mid)] for mid in self._ordered_ids()]

    @property
    def size(self) -> int:
        return int(self.index.ntotal)
