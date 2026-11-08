"""Interactive CLI, Automated Demo, and Diagnostic/Benchmark Runner for JARVIS v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .agent import JarvisAgent
from .config import JarvisConfig, MemoryConfig
from .eval_benchmark import run_full_verification, run_ir_benchmark


def run_demo(memory_dir: str = "jarvis_data/demo_memory") -> None:
    """Run an automated multi-turn demonstration with real-time token streaming."""
    cfg = JarvisConfig(memory=MemoryConfig(base_dir=Path(memory_dir)))
    agent = JarvisAgent(config=cfg)
    agent.memory_store.reset()

    print("=" * 76)
    print(f"JARVIS v2 Initialized | Model: {cfg.llm.model_name}")
    print(f"4-bit GGUF Footprint : {agent.llm_engine.get_memory_footprint_mb():.1f} MB")
    print(f"Embedding Model      : {cfg.memory.embedding_model} ({agent.memory_store.embedding_dim}-d)")
    print(f"Cross-Encoder Active : {agent.memory_store.reranker is not None} ({cfg.memory.reranker_model})")
    print("=" * 76)

    history: List[Dict[str, str]] = []
    demo_prompts = [
        "Hi, my name is Sparsh Agarwal, I am 20 years old, and I'm building a personal AI assistant called JARVIS connected to ROS 2 and robotics.",
        "What time is it right now, and calculate 128 * 37?",
        "What am I building, and what are my long-term robotics goals?",
        "whatismyage?",
        "What information do you know about me?",
    ]

    for idx, prompt in enumerate(demo_prompts, 1):
        print(f"\n[TURN {idx}] YOU: {prompt}")
        print(f"[TURN {idx}] JARVIS: ", end="", flush=True)

        tools_used: List[str] = []
        new_memories: List[Dict] = []
        metrics: Dict = {}

        for event in agent.stream_turn(prompt, history=history):
            ev_type = event.get("event")
            if ev_type == "tool_start":
                tools_used.append(event["tool_name"])
            elif ev_type == "token":
                print(event["text"], end="", flush=True)
            elif ev_type == "final":
                new_memories = event.get("new_memories", [])
                metrics = event.get("metrics", {})
            elif ev_type == "error":
                print(f"[Error: {event.get('message')}]", end="", flush=True)
        print()

        if tools_used:
            print(f"  -> Tools executed     : {', '.join(tools_used)}")
        if new_memories:
            print(f"  -> Extracted memories : {json.dumps(new_memories)}")
        if metrics:
            print(
                f"  -> Stream metrics     : TTFT={metrics.get('ttft_ms')}ms | "
                f"{metrics.get('tokens_per_sec')} tok/s | "
                f"tokens={metrics.get('output_tokens')}"
            )

    agent.memory_store.flush()
    print("\n" + "=" * 76)
    print("KV-Cache Telemetry:", json.dumps(agent.llm_engine.kv_manager.stats()))
    print(f"Total Persistent Memories ({agent.memory_store.size}):")
    for item in agent.memory_store.list_all():
        print(
            f"  [{item['id']}] ({item['memory_type']}, imp={item['importance']:.2f}, "
            f"access={item['access_count']}) {item['text']}"
        )
    print("=" * 76)


def run_interactive(
    memory_dir: str = "jarvis_data/memory", stream_output: bool = True
) -> None:
    """Start an interactive terminal session with word-by-word token streaming."""
    cfg = JarvisConfig(memory=MemoryConfig(base_dir=Path(memory_dir)))
    agent = JarvisAgent(config=cfg)
    history: List[Dict[str, str]] = []

    print("=" * 72)
    print(
        f"JARVIS v2 Online ({agent.llm_engine.get_memory_footprint_mb():.0f} MB GGUF | "
        f"{agent.memory_store.size} memories loaded | streaming={'on' if stream_output else 'off'})"
    )
    print("Commands: /memories, /kv, /tools, /metrics, /reset, /quit")
    print("=" * 72)

    try:
        while True:
            user_input = input("\nYOU: ").strip()
            if not user_input:
                continue
            cmd = user_input.lower()
            if cmd in {"/quit", "/exit", "quit", "exit"}:
                agent.memory_store.flush()
                print("JARVIS: Shutting down. Goodbye!")
                break
            if cmd == "/memories":
                for m in agent.memory_store.list_all():
                    print(
                        f"  [{m['id']}] {m['memory_type']} (imp={m['importance']:.2f}, "
                        f"access={m['access_count']}): {m['text']}"
                    )
                continue
            if cmd == "/kv":
                print(
                    "KV-Cache Stats:",
                    json.dumps(agent.llm_engine.kv_manager.stats(), indent=2),
                )
                continue
            if cmd == "/metrics":
                print(
                    "Last Stream Metrics:",
                    json.dumps(agent.llm_engine.last_stream_metrics, indent=2),
                )
                continue
            if cmd == "/tools":
                for s in agent.tools.schemas:
                    print(f"  - {s['name']}: {s['description']}")
                continue
            if cmd == "/reset":
                agent.memory_store.reset()
                agent.llm_engine.kv_manager.clear()
                history.clear()
                print("JARVIS: Memory store and KV-cache cleared.")
                continue

            if not stream_output:
                out = agent.turn(user_input, history=history)
                print(f"JARVIS: {out['answer']}")
                continue

            print("JARVIS: ", end="", flush=True)
            for event in agent.stream_turn(user_input, history=history):
                ev_type = event.get("event")
                if ev_type == "tool_start":
                    print(f"[{event['tool_name']}...] ", end="", flush=True)
                elif ev_type == "token":
                    # Prints each token/word immediately as generated
                    print(event["text"], end="", flush=True)
                elif ev_type == "error":
                    print(f"\n[Error: {event.get('message')}]", end="", flush=True)
            print()
    except (KeyboardInterrupt, EOFError):
        agent.memory_store.flush()
        print("\nJARVIS: Session saved. Goodbye!")


def main() -> None:
    parser = argparse.ArgumentParser(description="JARVIS v2 Local AI Assistant")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run automated multi-turn streaming demonstration.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run the 25-query IR benchmark and end-to-end verification suite.",
    )
    parser.add_argument(
        "--ir-only",
        action="store_true",
        help="Run only the fast 25-query IR benchmark (without loading the LLM).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable incremental token streaming in interactive CLI mode.",
    )
    parser.add_argument(
        "--memory-dir",
        type=str,
        default="jarvis_data/memory",
        help="Directory for persistent FAISS + metadata storage.",
    )
    args = parser.parse_args()

    if args.ir_only:
        res = run_ir_benchmark()
        print(json.dumps(res, indent=2))
    elif args.eval:
        res = run_full_verification(run_llm_e2e=True)
        print(json.dumps(res, indent=2))
    elif args.demo:
        run_demo()
    else:
        run_interactive(
            memory_dir=args.memory_dir, stream_output=not args.no_stream
        )


if __name__ == "__main__":
    main()
