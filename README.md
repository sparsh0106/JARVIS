# 🤖 J.A.R.V.I.S — Local Memory-Augmented AI & Dynamic Computer Agent

A modular, fully local AI assistant built on **`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`** (`llama-cpp-python`), **dynamic capability-oriented tool execution with deterministic policy validation**, **hybrid FAISS + TF-IDF retrieval**, **`BAAI/bge-reranker-base` Cross-Encoder reranking**, **query-aware memory selection with MMR diversity**, **Personal & Prefix KV-Cache retrieval**, and **real-time word-by-word streaming generation**.

---

## 📂 Project Structure

```text
JARVIS/
├── run_jarvis.py            # CLI entry point (--demo, --eval, --ir-only, or interactive)
├── requirements.txt         # Lightweight runtime Python dependencies
├── jarvis/
│   ├── __init__.py          # Public package exports
│   ├── __main__.py          # Entry point for `python -m jarvis`
│   ├── config.py            # Typed configuration (MemoryConfig, LLMConfig, AgentConfig)
│   ├── llm.py               # LLMBackend + Qwen3-4B GGUF loader, KV-cache & real-time token streaming
│   ├── kv_memory.py         # Personal KV-Cache & longest-common-prefix LRU tracker
│   ├── memory.py            # Hybrid FAISSMemoryStore (IndexIDMap2.reconstruct, TF-IDF, Cross-Encoder, MMR)
│   ├── tools.py             # Capability ToolRegistry (execute_command, filesystem, desktop, memory, calculate) & CommandPolicyValidator
│   ├── extractor.py         # Non-greedy raw_decode JSON parser & user-only durable memory extractor
│   ├── agent.py             # Proactive memory injection + bounded tool loop + token-by-token streaming
│   ├── eval_benchmark.py    # 25-query IR benchmark, dynamic capability tool policy checks, & E2E verification
│   └── cli.py               # Interactive terminal UI (incremental word-by-word streaming), demo, & benchmark runner
```

---

## 🛠️ Capability Tool Interface & Security Policy

Instead of one hard-coded tool per hardware metric or shell command, `ToolRegistry` exposes a small, capability-oriented interface validated by `CommandPolicyValidator`:

| Tool | Scope |
|---|---|
| `execute_command` | Runs a structured `executable` + `args` list with `shell=False`, sanitized environment variables, workspace `cwd` confinement, output truncation, and timeout bounds (`nvidia-smi`, `df`, `free`, `ps`, `lscpu`, `date`, `git`, `python3`/`pytest`, `ros2`). |
| `filesystem` | Workspace-scoped `list`, `search`, `read`, `write`, `edit`, `mkdir`, `copy`, `move`, and `delete` operations verified against configured `workspace_roots`. |
| `desktop` | Approved desktop operations (`open_app`, `list_apps`, `screenshot`). |
| `memory` | Unified `remember`, `search`, `update`, `forget`, and `list` operations backed by `FAISSMemoryStore`. |
| `calculate` | Bounded deterministic AST calculator with power-tower DoS protection. |

### Permission Levels
- **Read-only**: Hardware/system inspection (`nvidia-smi`, `ps`, `df`, `free`, `date`, `git status`, `ros2 topic list`) runs automatically after policy validation.
- **Reversible**: Workspace file creation/movement or `git add`/`branch` runs within approved workspace roots.
- **Destructive**: File deletion (`rm`, `filesystem delete`), `git reset`, package installation, and ROS 2 actuation (`ros2 topic pub`) require explicit confirmation (`confirm=True`).
- **Denied**: Privilege escalation (`sudo`, `su`), unrestricted shells (`bash -c`, `sh -c`), inline code execution (`python -c`), command chaining/injection, and path traversal outside `workspace_roots` are rejected deterministically.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch interactive JARVIS terminal chat (streams words/tokens in real time)
python run_jarvis.py

# 3. Run the automated multi-turn demonstration
python run_jarvis.py --demo

# 4. Run the 25-query IR benchmark + capability tool policy & E2E verification suite
python run_jarvis.py --eval
```

---

## 👤 Author

**Sparsh Agarwal** — [GitHub](https://github.com/sparsh0106) · [LinkedIn](https://www.linkedin.com/in/sparsh-agarwal-4b443b304/)
