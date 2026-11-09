"""Robust JSON parsing and Five-Category User Memory Extractor (no assistant laundering)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .llm import LocalLLMEngine
from .memory import VALID_MEMORY_TYPES

MEMORY_SYSTEM_PROMPT = (
    "You are the memory extraction engine for JARVIS. Your task is to extract durable, user-specific memories from the USER's statement and classify each into exactly one of five categories.\n\n"
    "Return ONLY a single valid JSON object in this exact format:\n"
    '{"memories": [{"text": "concise factual memory", "memory_type": "semantic", "importance": 0.85}]}\n'
    'If the statement contains no durable memories worth storing, return:\n'
    '{"memories": []}\n\n'
    "THE FIVE MEMORY CATEGORIES:\n"
    "1. semantic: Stable, enduring facts, knowledge, specs, and environment details about the user, their hardware, software stack, tools, environment, or system configuration.\n"
    '   - Examples:\n'
    '     * "My development laptop has an NVIDIA RTX 3060 Laptop GPU with 6 GB of VRAM." -> {"text": "The user\'s development laptop has an NVIDIA RTX 3060 Laptop GPU with 6 GB of VRAM.", "memory_type": "semantic", "importance": 0.90}\n'
    '     * "The JARVIS memory store uses FAISS for dense vector search." -> {"text": "The JARVIS memory store uses FAISS for dense vector search.", "memory_type": "semantic", "importance": 0.85}\n'
    '     * "The computer is running Ubuntu Linux." -> {"text": "The computer is running Ubuntu Linux.", "memory_type": "semantic", "importance": 0.85}\n'
    '     * "My main JARVIS project is written in Python." -> {"text": "The user\'s main JARVIS project is written in Python.", "memory_type": "semantic", "importance": 0.90}\n'
    "   - Boundary: Represents continuing facts/state (what IS), NOT reusable how-to steps (procedural) and NOT past events (episodic).\n\n"
    "2. procedural: Recurring methods, workflows, instructions, step-by-step procedures, commands, or operational rules for accomplishing tasks.\n"
    '   - Examples:\n'
    '     * "To activate the development environment, run conda activate torch_env." -> {"text": "To activate the development environment, run conda activate torch_env.", "memory_type": "procedural", "importance": 0.85}\n'
    '     * "To inspect GPU memory on this Linux machine, run nvidia-smi with a memory query." -> {"text": "To inspect GPU memory on this Linux machine, run nvidia-smi with a memory query.", "memory_type": "procedural", "importance": 0.80}\n'
    '     * "To run the unit tests, execute pytest from the repository root." -> {"text": "To run the unit tests, execute pytest from the repository root.", "memory_type": "procedural", "importance": 0.85}\n'
    '     * "To rebuild the ROS 2 workspace, source the ROS installation and run colcon build." -> {"text": "To rebuild the ROS 2 workspace, source the ROS installation and run colcon build from the workspace root.", "memory_type": "procedural", "importance": 0.85}\n'
    "   - Boundary: Action-oriented recipes ('To do X, execute Y'). NOT a past action (episodic) and NOT static specs (semantic).\n\n"
    "3. episodic: Specific past events, historical occurrences, debug sessions, past tests, or benchmark experiences anchored to a specific past time.\n"
    '   - Examples:\n'
    '     * "Yesterday I ran the JARVIS retrieval benchmark and recorded the Recall at 1 result." -> {"text": "The user ran the JARVIS retrieval benchmark yesterday and recorded Recall at 1.", "memory_type": "episodic", "importance": 0.75}\n'
    '     * "This morning I fixed an import error caused by running a package module as a script." -> {"text": "The user fixed an import error this morning caused by running a package module as a script.", "memory_type": "episodic", "importance": 0.70}\n'
    '     * "Last week I tested the local Qwen model with 4-bit quantization." -> {"text": "The user tested the local Qwen model with 4-bit quantization last week.", "memory_type": "episodic", "importance": 0.75}\n'
    '     * "During debugging, I discovered that the ROS 2 environment was not sourced in the process." -> {"text": "During debugging, the user discovered that the ROS 2 environment was not sourced.", "memory_type": "episodic", "importance": 0.70}\n'
    "   - Boundary: Anchored to a past occurrence ('Yesterday...', 'This morning...', 'Last week...', 'I completed a test...', 'During...'). NOT a general instruction (procedural).\n\n"
    "4. goal: Future intentions, plans, project milestones, objectives, or things the user wants to build, train, deploy, or achieve.\n"
    '   - Examples:\n'
    '     * "I want to connect JARVIS to my ROS 2 rover so it can inspect the robot." -> {"text": "The user wants to connect JARVIS to their ROS 2 rover.", "memory_type": "goal", "importance": 0.90}\n'
    '     * "My next project milestone is to add reliable streaming responses to JARVIS." -> {"text": "The user\'s next milestone is to add reliable streaming responses to JARVIS.", "memory_type": "goal", "importance": 0.85}\n'
    '     * "I plan to train a small vision model for detecting defects on a conveyor belt." -> {"text": "The user plans to train a small vision model for defect detection.", "memory_type": "goal", "importance": 0.85}\n'
    "   - Boundary: Future aspirations ('I want to...', 'I plan to...', 'My goal is...', 'My next milestone is...'). NOT subjective style choices (preference).\n\n"
    "5. preference: Subjective tastes, likes, dislikes, habits, style choices, opinionated development rules or constraints.\n"
    '   - Examples:\n'
    '     * "I prefer running language models locally instead of sending every request to a cloud API." -> {"text": "The user prefers running language models locally over cloud APIs.", "memory_type": "preference", "importance": 0.85}\n'
    '     * "I like Python examples that include type hints." -> {"text": "The user likes Python examples that include type hints.", "memory_type": "preference", "importance": 0.75}\n'
    '     * "I dislike desktop tools that execute commands without showing what they will do." -> {"text": "The user dislikes unconfirmed command execution.", "memory_type": "preference", "importance": 0.85}\n'
    '     * "I want potentially destructive computer actions to require my approval." -> {"text": "The user wants potentially destructive computer actions to require explicit approval.", "memory_type": "preference", "importance": 0.90}\n'
    "   - Boundary: Subjective preferences ('I prefer...', 'I like...', 'I dislike...'). NOT concrete project milestones (goal).\n\n"
    "WHAT NOT TO STORE (return {\"memories\": []}):\n"
    "- General knowledge or technical questions: 'What is the difference between FAISS and ChromaDB?', 'Explain how a Transformer computes self-attention.', 'What is the capital of Japan?'\n"
    "- Assistant task commands: 'Please calculate 128 multiplied by 37.', 'Find recent papers about visual navigation.', 'Can you solve this arithmetic expression?'\n"
    "- Transient physical or emotional states: 'I am hungry right now.', 'I am tired today.'\n"
    "- Hypothetical or counterfactual thoughts: 'Suppose I move to Germany next year.', 'Imagine that I buy a new workstation.', 'What would happen if a robot lost GPS reception?'\n"
    "- Greetings, conversational acknowledgments, or pleasantries: 'Hello, how are you?', 'Good night.', 'That explanation was useful, thank you.'\n\n"
    "EXTRACTION RULES:\n"
    "1. Return ONLY the JSON object. Do NOT wrap in markdown code blocks, do not explain.\n"
    "2. 'memory_type' MUST be exactly one of: semantic, procedural, episodic, goal, preference.\n"
    "3. 'importance' must be a float between 0.0 and 1.0.\n"
    "4. For semantic, goal, preference, and episodic, phrase from the user's perspective (e.g. 'The user...'). For procedural instructions, preserve the command recipe (e.g. 'To activate..., run...')."
)

_TRANSIENT_OR_IRRELEVANT_PATTERNS = [
    # Pure greetings, closings, and pleasantries
    re.compile(r"^\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening|night)|bye|goodbye|see\s+ya)[!.]*\s*$", re.I),
    # Pure conversational acknowledgments & gratitude
    re.compile(r"^\s*(?:thanks?|thank\s+you|that\s+(?:explanation\s+)?was\s+(?:useful|helpful|great)|ok|okay|cool|great|awesome|sure|no\s+problem)[!.]*\s*$", re.I),
    # Transient physical or emotional feelings (fleeting, not durable)
    re.compile(r"^\s*(?:i\s+am|i'm)\s+(?:hungry|thirsty|tired|sleepy|bored|cold|hot)\b", re.I),
    # Hypothetical or counterfactual thought experiments
    re.compile(r"^\s*(?:suppose\b|imagine\s+(?:that)?\b|what\s+would\s+happen\s+if\b)", re.I),
    # Direct assistant commands without personal disclosure
    re.compile(r"^\s*(?:please\s+)?(?:calculate|compute|solve|find\s+recent|summarize\s+(?:this|the)|explain\s+(?:how|the|concept)|tell\s+me\s+how|can\s+you\s+(?:solve|summarize|give\s+me|explain))\b", re.I),
    # Pure general informational questions (starting with question words and ending with ?)
    re.compile(r"^\s*(?:what\s+is|what\s+are|what\s+was|how\s+does|how\s+do|how\s+can|why\s+is|why\s+do|where\s+is|when\s+did|who\s+is|can\s+you|could\s+you)\b.*\?\s*$", re.I),
]


def _repair_common_json_typos(text: str) -> str:
    """Repair common small-model JSON typos such as stray quotes before closing braces."""
    repaired = re.sub(r'\]\s*"\s*\}', "]}", text)
    return re.sub(r'\}\s*"\s*\]', "}]", repaired)


def extract_all_json_objects(text: str) -> List[Dict[str, Any]]:
    """Extract all top-level JSON objects from text in order using `raw_decode`."""
    if not text:
        return []

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    cleaned = _repair_common_json_typos(cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    found: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(cleaned):
        brace_pos = cleaned.find("{", idx)
        if brace_pos == -1:
            break
        try:
            obj, end_pos = decoder.raw_decode(cleaned, brace_pos)
            if isinstance(obj, dict):
                found.append(obj)
                idx = end_pos
                continue
        except json.JSONDecodeError:
            pass
        idx = brace_pos + 1

    return found


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse the first valid JSON object from `text` non-greedily."""
    objs = extract_all_json_objects(text)
    return objs[0] if objs else None


def is_declarative_user_statement(user_text: str) -> bool:
    """Lightweight filter: passes statements to the LLM unless obviously transient, conversational, or irrelevant."""
    t = user_text.strip()
    if len(t) < 4:
        return False
    for pat in _TRANSIENT_OR_IRRELEVANT_PATTERNS:
        if pat.search(t):
            return False
    return True


def extract_memories_from_user_turn(
    user_text: str,
    llm_engine: LocalLLMEngine,
) -> List[Dict[str, Any]]:
    """Extract durable user-specific memories strictly from `user_text` (never from assistant replies)."""
    if not is_declarative_user_statement(user_text):
        return []

    prompt = (
        f"USER STATEMENT:\n{user_text.strip()}\n\n"
        "Extract atomic durable user-specific memories from the USER STATEMENT above."
    )

    raw = llm_engine.generate_chat(
        [
            {"role": "system", "content": MEMORY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_new_tokens=240,
        temperature=0.0,
        do_sample=False,
    )

    parsed = extract_json_object(raw)
    if not parsed:
        return []

    memories: List[Any] = (
        parsed["memories"]
        if isinstance(parsed.get("memories"), list)
        else [obj for obj in extract_all_json_objects(raw) if "text" in obj]
    )

    validated: List[Dict[str, Any]] = []
    for item in memories:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text or re.match(r"^(?:your\s+|you\s+could\s+|you\s+should\s+|let\s+me\s+know)", text, re.I):
            continue

        memory_type = item.get("memory_type")
        if memory_type not in VALID_MEMORY_TYPES:
            memory_type = "semantic"

        try:
            importance = max(0.0, min(1.0, float(item.get("importance", 0.75))))
        except (TypeError, ValueError):
            importance = 0.75

        validated.append({
            "text": text,
            "memory_type": memory_type,
            "importance": importance,
        })

    return validated
