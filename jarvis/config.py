"""Configuration specification for the modular JARVIS v2 architecture."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_GGUF_PATH = (
    "/home/sparsh/.lmstudio/models/lmstudio-community/"
    "Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
)


@dataclass
class MemoryConfig:
    """Configuration for the FAISS + TF-IDF + Cross-Encoder Memory System."""

    base_dir: Path = field(default_factory=lambda: Path("jarvis_data/memory"))
    embedding_model: str = "sentence-transformers/all-MiniLM-L12-v2"
    reranker_model: str = "BAAI/bge-reranker-base"
    use_cross_encoder: bool = True
    cross_encoder_weight: float = 0.55

    # Candidate retrieval & fusion parameters
    candidate_k: int = 50
    rrf_k: int = 60
    tfidf_weight: float = 0.35
    dense_weight: float = 0.65

    # Final ranking weights (applied ONLY after passing relevance gating)
    alpha_relevance: float = 0.72
    beta_importance: float = 0.14
    gamma_recency: float = 0.09
    delta_access: float = 0.05

    # Decay, thresholds, and diversity
    recency_half_life_days: float = 30.0
    min_relevance_threshold: float = 0.22
    duplicate_threshold: float = 0.92
    supersede_threshold: float = 0.82
    mmr_lambda: float = 0.78

    # I/O flush policy: avoid rewriting disk on every read query
    flush_every_n_reads: int = 10


@dataclass
class LLMConfig:
    """Configuration for local 4-bit GGUF LLM (Qwen3-4B) and KV-Cache management."""

    model_name: str = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    model_path: str = field(
        default_factory=lambda: os.getenv("JARVIS_GGUF_PATH", DEFAULT_GGUF_PATH)
    )
    n_ctx: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_N_CTX", "8192"))
    )
    n_gpu_layers: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_N_GPU_LAYERS", "-1"))
    )
    load_in_4bit: bool = True
    max_new_tokens: int = 512
    max_tokens_limit: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    verbose: bool = False
    use_prefix_kv_cache: bool = True
    use_personal_kv_cache: bool = True
    max_kv_checkpoints: int = 32


@dataclass
class AgentConfig:
    """Configuration for the JARVIS Agent Loop, Capability Tools, and Execution Policy."""

    max_tool_rounds: int = 6
    max_tool_retries: int = 2
    auto_inject_memories: bool = True
    auto_inject_top_k: int = 4
    auto_extract_memories: bool = True
    keep_json_in_history: bool = False

    # Workspace & execution policy configuration
    workspace_root: Path = field(default_factory=Path.cwd)
    workspace_roots: Tuple[Path, ...] = field(default_factory=tuple)
    command_timeout_seconds: int = 10
    max_command_timeout_seconds: int = 30
    max_output_chars: int = 4000
    require_confirmation_for_destructive: bool = True
    allow_app_launch: bool = True


@dataclass
class JarvisConfig:
    """Top-level JARVIS system configuration."""

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    device: Optional[str] = None
