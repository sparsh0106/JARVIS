"""Robust JSON parsing and User-Only Durable Memory Extractor (no assistant laundering)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .llm import LocalLLMEngine
from .memory import VALID_MEMORY_TYPES

MEMORY_SYSTEM_PROMPT = (
    "You extract atomic, durable user-specific memories strictly from the USER's statement.\n"
    "Return ONLY valid JSON in this exact format:\n"
    '{"memories":['
    '{"text":"The user\'s name is ...","memory_type":"semantic","importance":0.95},'
    '{"text":"The user\'s age is ... years old","memory_type":"semantic","importance":0.95},'
    '{"text":"The user wants to ...","memory_type":"goal","importance":0.90}'
    ']}\n\n'
    "Valid memory_type values: episodic, semantic, procedural, preference, goal.\n"
    "Rules:\n"
    "1. Extract EVERY distinct personal fact mentioned (name, age, profession, project, long-term goal, preference) as a separate item.\n"
    "2. Extract ONLY facts, goals, preferences, projects, or personal details explicitly stated by the USER.\n"
    "3. Write each memory as a concise third-person statement starting with 'The user ...'.\n"
    "4. If the USER is only asking a question, greeting, or issuing a utility command, return {\"memories\":[]}.\n"
    "5. Do NOT include any assistant-generated text or inferred information; extract only what the USER explicitly states.\n"
    "6. Assign an importance score between 0.0 and 1.0 for each memory, reflecting its significance to the USER's identity or goals.\n"
    "7. If the USER provides multiple facts in one statement, extract each fact as a separate memory object.\n"
    "8. Output valid JSON only."
)

_DECLARATIVE_USER_CUES = re.compile(
    r"\b(?:i\s+am|i'm|im|my\s+\w+\s+is|i\s+want|i\s+like|i\s+prefer|i\s+love|i\s+hate|"
    r"i\s+work|i\s+study|i\s+build|i'm\s+building|i\s+plan|i\s+need\s+to|my\s+goal|"
    r"remember\s+that|call\s+me|years?\s+old)\b",
    re.I,
)


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

    # Fast path: entire string is valid JSON
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
    """Return True only if `user_text` contains a personal fact, goal, or preference worth extracting."""
    t = user_text.strip()
    return len(t) >= 6 and bool(_DECLARATIVE_USER_CUES.search(t))


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
