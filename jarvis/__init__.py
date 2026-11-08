"""JARVIS v2: Local Memory-Augmented AI & Dynamic Computer Assistant."""

from .agent import JarvisAgent, build_streaming_system_prompt, build_system_prompt
from .config import AgentConfig, JarvisConfig, LLMConfig, MemoryConfig
from .extractor import extract_json_object, extract_memories_from_user_turn
from .kv_memory import PersonalKVCacheManager
from .llm import LLMBackend, LocalLLMEngine
from .memory import FAISSMemoryStore, QueryAwareRouter, VALID_MEMORY_TYPES
from .tools import (
    CommandPolicyValidator,
    CommandRequest,
    CommandResult,
    PolicyDecision,
    ToolRegistry,
    calculate,
)

__all__ = [
    "AgentConfig",
    "CommandPolicyValidator",
    "CommandRequest",
    "CommandResult",
    "FAISSMemoryStore",
    "JarvisAgent",
    "JarvisConfig",
    "LLMBackend",
    "LLMConfig",
    "LocalLLMEngine",
    "MemoryConfig",
    "PersonalKVCacheManager",
    "PolicyDecision",
    "QueryAwareRouter",
    "ToolRegistry",
    "VALID_MEMORY_TYPES",
    "build_streaming_system_prompt",
    "build_system_prompt",
    "calculate",
    "extract_json_object",
    "extract_memories_from_user_turn",
]
