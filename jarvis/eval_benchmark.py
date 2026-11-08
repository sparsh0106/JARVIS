"""25-Query IR Benchmark (Recall@K, MRR, Negative Gating), Dynamic Capability Tool Policy Tests, and End-to-End Verification."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from .agent import JarvisAgent, build_system_prompt
from .api.llm_service import LocalLLMService
from .api.server import _validate_agent_stream_payload
from .config import JarvisConfig, MemoryConfig
from .extractor import (
    MEMORY_SYSTEM_PROMPT,
    extract_json_object,
    is_declarative_user_statement,
)
from .memory import FAISSMemoryStore, QueryAwareRouter
from .tools import ToolRegistry, calculate

# Isolated test-only corpus (never touches production jarvis_data/)
BENCHMARK_CORPUS: List[Tuple[str, str, float]] = [
    ("The user's name is Sparsh Agarwal.", "semantic", 0.95),
    ("The user's age is 20 years old.", "semantic", 0.99),
    ("The user is studying computer science and wants to be an AI and robotics engineer.", "semantic", 0.85),
    ("The user is building a personal AI assistant called JARVIS that runs locally on an RTX 3060.", "semantic", 0.95),
    ("The user wants JARVIS eventually connected to ROS 2 for mobile robot navigation and control.", "goal", 0.92),
    ("The user's long-term goal is to integrate vision-language-action models with robotic manipulators.", "goal", 0.90),
    ("The user prefers dark mode themes in VS Code and terminal emulators.", "preference", 0.70),
    ("The user prefers writing Python with PyTorch and strict type annotations.", "preference", 0.80),
    ("The user dislikes cloud-only AI assistants due to latency and privacy concerns.", "preference", 0.75),
    ("To launch the local development environment, activate the conda environment named torch_env.", "procedural", 0.85),
    ("To run the robot simulation, source /opt/ros/humble/setup.bash and run ros2 launch.", "procedural", 0.88),
    ("Last week the user debugged a FAISS dimension mismatch caused by switching embedding models.", "episodic", 0.65),
    ("Yesterday the user benchmarked Qwen3-1.7B with 4-bit NF4 quantization using BitsAndBytes.", "episodic", 0.75),
    ("The user enjoys Formula 1 racing and follows lap telemetry analysis.", "preference", 0.60),
    ("The user stores machine learning research papers in the Papers directory.", "procedural", 0.65),
]

# 20 positive retrieval queries -> expected 1-indexed memory ID in BENCHMARK_CORPUS
POSITIVE_EVAL_QUERIES: List[Tuple[str, int, Optional[str]]] = [
    ("What is my full name?", 1, None),
    ("Who am I?", 1, None),
    ("How old am I?", 2, None),
    ("whatismyage?", 2, None),
    ("what is sparsh agarwal's age?", 2, None),
    ("What career or engineering field am I pursuing?", 3, None),
    ("What am I building and what GPU does it run on?", 4, None),
    ("How does ROS 2 fit into my long-term JARVIS goals?", 5, "goal"),
    ("What are my goals for robotic manipulators and VLA models?", 6, "goal"),
    ("What UI theme do I prefer in my code editor?", 7, "preference"),
    ("Which deep learning framework and coding style do I prefer?", 8, "preference"),
    ("Why do I avoid cloud-based AI services?", 9, "preference"),
    ("Which conda environment should I activate for development?", 10, "procedural"),
    ("How do I launch the ROS 2 robot simulation?", 11, "procedural"),
    ("What vector database error did I fix last week?", 12, "episodic"),
    ("Which language model quantization did I test yesterday?", 13, "episodic"),
    ("What motorsport do I enjoy watching?", 14, "preference"),
    ("Where do I keep my AI research PDFs?", 15, "procedural"),
    ("Tell me about my personal assistant project JARVIS.", 4, None),
    ("What are my robotics navigation plans?", 5, None),
]

# 5 negative/unrelated queries that should return NO personal memories via query-aware routing or threshold gating
NEGATIVE_EVAL_QUERIES: List[str] = [
    "What time is it?",
    "Calculate 128 * 37",
    "What is today's date?",
    "Hello!",
    "999 * 123 + 45",
]


def _verify_api_payload_validation() -> bool:
    """Verify FastAPI payload validation rules (HTTP 400 and 422 checks)."""
    svc = LocalLLMService(max_tokens_limit=2048)

    # 1. Non-object body -> 400
    try:
        svc._validate_payload(["not", "a", "dict"])
        return False
    except HTTPException as exc:
        if exc.status_code != 400:
            return False

    # 2. Missing messages -> 422
    try:
        svc._validate_payload({"stream": True})
        return False
    except HTTPException as exc:
        if exc.status_code != 422:
            return False

    # 3. Invalid stream type -> 422
    try:
        svc._validate_payload(
            {"messages": [{"role": "user", "content": "Hi"}], "stream": "yes"}
        )
        return False
    except HTTPException as exc:
        if exc.status_code != 422:
            return False

    # 4. Out-of-bounds max_tokens -> 422
    try:
        svc._validate_payload(
            {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 9999}
        )
        return False
    except HTTPException as exc:
        if exc.status_code != 422:
            return False

    # 5. Valid payload strips client 'model' field
    cleaned = svc._validate_payload(
        {
            "model": "/etc/passwd",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
    )
    if "model" in cleaned or cleaned.get("stream") is not True:
        return False

    # 6. Agent stream payload validation
    agent_params = _validate_agent_stream_payload(
        {"user_text": "What is my age?", "history": []}
    )
    return agent_params["user_text"] == "What is my age?"


def _verify_dynamic_capability_tools(store: FAISSMemoryStore, tmp_dir: Path) -> Dict[str, bool]:
    """Verify dynamic command execution policy, workspace filesystem isolation, desktop, and memory tools."""
    ws = tmp_dir / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    tools = ToolRegistry(
        memory_store=store,
        workspace_root=ws,
        allow_app_launch=False,
        require_confirmation_for_destructive=True,
    )

    results: Dict[str, bool] = {}

    # 1. Capability tool set migration check
    expected_tools = {"execute_command", "filesystem", "desktop", "memory", "calculate"}
    results["capability_tool_set_exact"] = tools.tool_names == expected_tools

    # 2. Read-only system commands execute automatically
    uname_res = tools.execute("execute_command", {"executable": "uname", "args": ["-s"]})
    date_res = tools.execute("execute_command", {"executable": "date", "args": []})
    py_res = tools.execute("execute_command", {"executable": "python3", "args": ["--version"]})
    results["readonly_commands_succeed"] = (
        bool(uname_res.get("ok"))
        and bool(date_res.get("ok"))
        and bool(py_res.get("ok"))
        and len(uname_res.get("stdout", "")) > 0
    )

    # 3. Reject unknown executables, shells, privilege escalation, inline python, and shell injection
    unknown_res = tools.execute("execute_command", {"executable": "nonexistent_malicious_bin", "args": []})
    sudo_res = tools.execute("execute_command", {"executable": "sudo", "args": ["ls"]})
    bash_res = tools.execute("execute_command", {"executable": "bash", "args": ["-c", "echo pwned"]})
    pyc_res = tools.execute("execute_command", {"executable": "python3", "args": ["-c", "import os"]})
    inj_res = tools.execute("execute_command", {"executable": "df", "args": ["-h; rm -rf /"]})
    trav_cmd_res = tools.execute("execute_command", {"executable": "cat", "args": ["/etc/passwd"]})
    results["security_policy_rejects_unapproved"] = all(
        r.get("ok") is False and r.get("status") == "denied"
        for r in (unknown_res, sudo_res, bash_res, pyc_res, inj_res, trav_cmd_res)
    )

    # 4. Destructive commands require explicit confirmation
    rm_unconfirmed = tools.execute("execute_command", {"executable": "rm", "args": ["test.txt"]})
    git_reset = tools.execute("execute_command", {"executable": "git", "args": ["reset", "--hard"]})
    ros_pub = tools.execute("execute_command", {"executable": "ros2", "args": ["topic", "pub", "/cmd_vel", "geometry_msgs/msg/Twist"]})
    results["destructive_commands_require_confirmation"] = all(
        r.get("ok") is False
        and r.get("status") == "confirmation_required"
        and r.get("requires_confirmation") is True
        for r in (rm_unconfirmed, git_reset, ros_pub)
    )

    # 5. Non-zero exit codes are accurately reported as failed (never ok=True)
    fail_ls = tools.execute("execute_command", {"executable": "ls", "args": ["nonexistent_file_xyz.txt"]})
    results["nonzero_exit_code_reported_failed"] = (
        fail_ls.get("ok") is False
        and fail_ls.get("status") == "failed"
        and fail_ls.get("exit_code") != 0
    )

    # 6. Filesystem capability operations and path traversal defense
    w_res = tools.execute("filesystem", {"operation": "write", "path": "notes/todo.txt", "content": "hello world\nline two"})
    r_res = tools.execute("filesystem", {"operation": "read", "path": "notes/todo.txt"})
    e_res = tools.execute("filesystem", {"operation": "edit", "path": "notes/todo.txt", "target": "world", "replacement": "jarvis"})
    r2_res = tools.execute("filesystem", {"operation": "read", "path": "notes/todo.txt"})
    s_res = tools.execute("filesystem", {"operation": "search", "path": ".", "pattern": "*.txt"})
    del_unconfirmed = tools.execute("filesystem", {"operation": "delete", "path": "notes/todo.txt"})
    del_confirmed = tools.execute("filesystem", {"operation": "delete", "path": "notes/todo.txt", "confirm": True})
    escape_res = tools.execute("filesystem", {"operation": "read", "path": "../../etc/passwd"})

    results["filesystem_operations_and_isolation_ok"] = (
        bool(w_res.get("ok"))
        and "hello world" in r_res.get("content", "")
        and bool(e_res.get("ok"))
        and "hello jarvis" in r2_res.get("content", "")
        and "notes/todo.txt" in s_res.get("matches", [])
        and del_unconfirmed.get("status") == "confirmation_required"
        and bool(del_confirmed.get("ok"))
        and escape_res.get("status") == "denied"
    )

    # 7. Consolidated memory capability tool
    m_rem = tools.execute("memory", {"operation": "remember", "text": "The user prefers Arch Linux.", "memory_type": "preference", "importance": 0.85})
    m_srch = tools.execute("memory", {"operation": "search", "query": "Which Linux distro does the user prefer?"})
    results["consolidated_memory_tool_ok"] = (
        bool(m_rem.get("ok"))
        and bool(m_srch.get("ok"))
        and "Arch Linux" in m_srch.get("results", "")
    )

    # 8. Desktop capability tool (dry-run & unapproved rejection)
    d_open = tools.execute("desktop", {"operation": "open_app", "app_name": "calculator"})
    d_bad = tools.execute("desktop", {"operation": "open_app", "app_name": "evil_miner"})
    results["desktop_tool_ok"] = (
        bool(d_open.get("ok")) and d_bad.get("status") == "denied"
    )

    return results


def run_ir_benchmark(
    embedder: Any = None,
    reranker: Any = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate Recall@1, Recall@3, MRR, and Negative Query Rejection across the 25-query benchmark,
    comparing Hybrid (Bi-Encoder + RRF) vs Hybrid + Cross-Encoder Reranking.
    """
    with tempfile.TemporaryDirectory(prefix="jarvis_ir_bench_") as tmp_dir:
        cfg = MemoryConfig(
            base_dir=Path(tmp_dir),
            use_cross_encoder=True,
            duplicate_threshold=0.95,
        )
        store = FAISSMemoryStore(
            config=cfg, embedder=embedder, reranker=reranker, device=device
        )

        for text, mtype, imp in BENCHMARK_CORPUS:
            store.add(
                text=text,
                memory_type=mtype,
                importance=imp,
                source="benchmark_seed",
            )

        modes = {
            "hybrid_biencoder_only": False,
            "hybrid_plus_cross_encoder": True,
        }
        report: Dict[str, Any] = {}

        for mode_name, use_ce in modes.items():
            hits_at_1 = 0
            hits_at_3 = 0
            rr_sum = 0.0

            for query, expected_id, mtype in POSITIVE_EVAL_QUERIES:
                results = store.hybrid_search(
                    query=query,
                    top_k=5,
                    memory_type=mtype,
                    use_reranker=use_ce,
                    use_mmr=True,
                    query_aware=True,
                    record_access=False,
                )
                ranked_ids = [r["id"] for r in results]
                if expected_id in ranked_ids:
                    rank = ranked_ids.index(expected_id) + 1
                    rr_sum += 1.0 / rank
                    if rank == 1:
                        hits_at_1 += 1
                    if rank <= 3:
                        hits_at_3 += 1

            n_pos = len(POSITIVE_EVAL_QUERIES)
            neg_rejected = 0
            for neg_q in NEGATIVE_EVAL_QUERIES:
                plan = QueryAwareRouter.analyze(neg_q, cfg)
                if not plan.needs_memory:
                    neg_rejected += 1
                else:
                    res = store.hybrid_search(
                        query=neg_q,
                        top_k=3,
                        use_reranker=use_ce,
                        record_access=False,
                    )
                    if not res:
                        neg_rejected += 1

            report[mode_name] = {
                "recall_at_1": round(hits_at_1 / n_pos, 4),
                "recall_at_3": round(hits_at_3 / n_pos, 4),
                "mrr": round(rr_sum / n_pos, 4),
                "negative_rejection_rate": round(
                    neg_rejected / len(NEGATIVE_EVAL_QUERIES), 4
                ),
            }

        # Verify pre-filtering when store has >50 items of another memory_type
        for i in range(55):
            store.add(
                text=f"Synthetic distractor semantic fact number {i} about ROS 2 robotics.",
                memory_type="semantic",
                importance=0.99,
                source="distractor",
                duplicate_threshold=0.999,
            )
        filtered_hits = store.hybrid_search(
            query="What vector database error did I fix last week?",
            top_k=3,
            memory_type="episodic",
            record_access=False,
        )
        report["prefilter_over_50_distractors_ok"] = (
            len(filtered_hits) >= 1 and filtered_hits[0]["id"] == 12
        )

    return report


def run_full_verification(run_llm_e2e: bool = True) -> Dict[str, Any]:
    """Run deterministic unit checks for all capability tools, memory retrieval, and live LLM streaming."""
    checks: Dict[str, Any] = {}

    # 1. Prompt newline verification (no literal "\\n")
    sample_sys = build_system_prompt('{"name":"execute_command"}')
    checks["system_prompt_has_real_newlines"] = (
        "\\n" not in sample_sys and "\n" in sample_sys
    )
    checks["memory_prompt_has_real_newlines"] = (
        "\\n" not in MEMORY_SYSTEM_PROMPT and "\n" in MEMORY_SYSTEM_PROMPT
    )

    # 2. Safe AST Calculator DoS protection against 9**9**9**9
    t0 = time.perf_counter()
    dos_out = calculate("9**9**9**9")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    checks["calculator_blocks_power_tower_dos"] = (
        "error" in dos_out.lower() and elapsed_ms < 50.0
    )
    checks["calculator_128_x_37"] = calculate("128 * 37") == "4736"

    # 3. Non-greedy JSON extraction on multiple JSON blocks
    multi_json = 'Here is the action:\n{"action":"final","response":"Hello Sparsh"}\n{"trailing":"ignored"}'
    parsed = extract_json_object(multi_json)
    checks["non_greedy_json_extraction_ok"] = (
        parsed is not None and parsed.get("response") == "Hello Sparsh"
    )

    # 4. Question vs Declarative filtering for memory extraction (prevents assistant echo)
    checks["question_skipped_by_extractor"] = not is_declarative_user_statement(
        "What are my longer-term goals for JARVIS?"
    )
    checks["declarative_accepted_by_extractor"] = is_declarative_user_statement(
        "My name is Sparsh Agarwal and I am building a ROS 2 robot."
    )

    # 5. FastAPI Streaming Payload Validation Checks (400 / 422 / model stripping)
    checks["api_payload_validation_ok"] = _verify_api_payload_validation()

    # 6. Dynamic Capability Tools + IR Benchmark + Singleton Supersede + Reset verification
    with tempfile.TemporaryDirectory(prefix="jarvis_verify_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = MemoryConfig(base_dir=tmp_path / "mem", use_cross_encoder=True)
        store = FAISSMemoryStore(config=cfg)
        id1 = store.add(
            "The user's age is 20 years old.",
            memory_type="semantic",
            importance=0.9,
        )
        id2 = store.add(
            "The user's age is 21 years old.",
            memory_type="semantic",
            importance=0.95,
        )
        checks["singleton_fact_supersedes_in_place"] = (
            id1 == id2
            and store.size == 1
            and "21" in store.list_all()[0]["text"]
        )
        store.reset()
        checks["reset_clears_ram_and_disk"] = (
            store.size == 0 and not store.index_path.exists()
        )

        checks["dynamic_capability_tools"] = _verify_dynamic_capability_tools(
            store, tmp_path
        )
        store.reset()

        ir_metrics = run_ir_benchmark(
            embedder=store.embedder,
            reranker=store.reranker,
            device=store.device,
        )
        checks["ir_benchmark"] = ir_metrics

        if run_llm_e2e:
            app_cfg = JarvisConfig(
                memory=MemoryConfig(base_dir=tmp_path / "e2e_mem")
            )
            agent = JarvisAgent(
                config=app_cfg,
                memory_store=FAISSMemoryStore(
                    config=app_cfg.memory,
                    embedder=store.embedder,
                    reranker=store.reranker,
                    device=store.device,
                ),
            )
            footprint_mb = agent.llm_engine.get_memory_footprint_mb()
            checks["llm_4bit_footprint_mb"] = round(footprint_mb, 1)
            checks["llm_4bit_quantization_applied"] = footprint_mb < 3500.0

            # Seed clean test facts
            agent.memory_store.add(
                "The user's name is Sparsh Agarwal and he is a robotics & AI programmer.",
                memory_type="semantic",
                importance=0.95,
                source="test_seed",
            )
            agent.memory_store.add(
                "The user's age is 20 years old.",
                memory_type="semantic",
                importance=0.99,
                source="test_seed",
            )
            agent.memory_store.add(
                "The user is building a personal AI assistant called JARVIS and wants to connect it to ROS 2 and robotics.",
                memory_type="goal",
                importance=0.95,
                source="test_seed",
            )

            history: List[Dict[str, str]] = []
            t_goals = agent.turn(
                "What am I building, and what are my long-term robotics goals?",
                history=history,
            )
            t_age = agent.turn("whatismyage?", history=history)
            t_calc = agent.turn("Calculate 128 * 37", history=history)

            checks["e2e_turn_goals_answer"] = t_goals["answer"]
            checks["e2e_turn_goals_no_echo_memories"] = (
                len(t_goals["new_memories"]) == 0
            )
            checks["e2e_turn_age_answer"] = t_age["answer"]
            checks["e2e_turn_age_correct"] = "20" in t_age["answer"]
            checks["e2e_turn_calc_answer"] = t_calc["answer"]
            checks["e2e_turn_calc_correct"] = "4736" in t_calc["answer"]

            # 7. Verify Agent Streaming (`stream_turn_sync`) events, no JSON leak, and TTFT telemetry
            stream_events = list(
                agent.stream_turn_sync(
                    "What is my name and what am I building?", history=history
                )
            )
            event_types = [e.get("event") for e in stream_events]
            token_texts = [
                e.get("text", "")
                for e in stream_events
                if e.get("event") == "token"
            ]
            streamed_answer = "".join(token_texts).strip()
            final_ev = next(
                (e for e in stream_events if e.get("event") == "final"), {}
            )

            checks["streaming_events_valid"] = (
                event_types[0] == "start"
                and event_types[-1] == "done"
                and "token" in event_types
                and "final" in event_types
            )
            checks["streaming_no_json_leak"] = not streamed_answer.startswith(
                "{"
            )
            checks["streaming_answer"] = streamed_answer
            checks["streaming_metrics"] = final_ev.get("metrics", {})
            checks["kv_cache_stats"] = agent.llm_engine.kv_manager.stats()

    return checks


if __name__ == "__main__":
    import json
    import sys

    run_llm = "--no-llm" not in sys.argv
    print(f"Running JARVIS verification suite (run_llm_e2e={run_llm})...")
    res = run_full_verification(run_llm_e2e=run_llm)
    print(json.dumps(res, indent=2))
