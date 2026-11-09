# JARVIS: Memory-Augmented Local Agent Architecture

JARVIS is a local, privacy-first personal AI assistant and capability agent. It couples 4-bit quantized GGUF model execution via `llama.cpp` with a hybrid vector and lexical memory subsystem (FAISS and TF-IDF) and sandboxed system execution tools. The architecture is designed to run within a 6 GB GPU VRAM envelope without external API dependencies.

## Authors

- Siddhi Pateriya
- Sparsh Agarwal

---

## Architecture Overview

JARVIS consists of four core subsystems:

```
+--------------------------------------------------------------------------+
|                               JARVIS Agent                               |
|   (Turn Context Preparation, Query Routing, Multi-Step Tool Loop)        |
+---------------------+--------------------+-------------------------------+
                      |                    |
        +-------------v------------+       |
        |    FAISSMemoryStore      |       |
        | - Dense Embeddings       |       |
        |   (all-MiniLM-L12-v2)    |       |
        | - Sparse Lexical         |       |
        |   (TF-IDF sublinear)     |       |
        | - Reciprocal Rank Fusion |       |
        | - Gated Memory Types     |       |
        +--------------------------+       |
                                           |
        +----------------------------------v-------------------------------+
        |                         LocalLLMEngine                           |
        | - 4-bit Quantized GGUF Model (Qwen3-4B-Instruct-2507-Q4_K_M)     |
        | - llama.cpp Runtime with KV-Cache State Management               |
        | - Word-by-Word Real-Time Streaming & Prefix Reuse                |
        +----------------------------------+-------------------------------+
                                           |
        +----------------------------------v-------------------------------+
        |                          ToolRegistry                            |
        | - execute_command (timed out, read-only policy check)            |
        | - filesystem (workspace-contained file read/write)               |
        | - calculate (AST-parsed safe arithmetic, bounded exponents)      |
        | - memory (remember, update, forget, inspect)                     |
        +------------------------------------------------------------------+
```

### 1. Hybrid Memory Subsystem (`jarvis.memory`)

The memory system supports five memory types: `goal`, `preference`, `semantic`, `procedural`, and `episodic`.

- **Dense Search**: `sentence-transformers/all-MiniLM-L12-v2` produces 384-dimensional normalized embeddings indexed in a `faiss.IndexIDMap2(faiss.IndexFlatIP)`.
- **Sparse Search**: `scikit-learn` `TfidfVectorizer` with `(1, 2)` n-grams and sublinear term frequency scaling for exact keyword and entity recall.
- **Fusion & Ranking**: Candidates from dense and sparse pools are merged via Reciprocal Rank Fusion (RRF with `k=60`). The final rank combines relevance with importance, exponential half-life recency decay, and logarithmic access frequency.
- **Dynamic Query Routing**: A query-aware router classifies user intent into memory retrieval plans, determining whether personal memory injection is needed and adjusting dense versus lexical weighting.
- **Optional Reranker**: Integrated support for `BAAI/bge-reranker-base` for pairwise Cross-Encoder score refinement.

### 2. Local LLM & KV-Cache Engine (`jarvis.llm`, `jarvis.kv_memory`)

- **Model Execution**: `llama-cpp-python` runs quantized GGUF weights directly on CUDA hardware.
- **Prefix Reuse**: Preserves static prompt prefixes (system directives and tool specifications) and multi-turn conversational history in the KV cache across consecutive requests, eliminating repeated prompt prefill passes.
- **Streaming Pipeline**: Generates word-by-word token deltas while tracking Time-To-First-Token (TTFT), tokens-per-second throughput, and KV-cache reuse statistics.

### 3. Capability Tools & Sandboxing (`jarvis.tools`)

- **`execute_command`**: Runs terminal commands under strict execution policies. Enforces configurable timeouts, character output caps, and read-only flags for destructive system commands.
- **`filesystem`**: Confines file creation, reading, appending, and directory listing within configured workspace boundaries.
- **`calculate`**: Evaluates mathematical expressions using Python Abstract Syntax Tree (AST) parsing with operator node whitelisting and bounded exponentiation to prevent denial-of-service attempts.
- **`memory`**: Allows the agent to explicitly store, inspect, update, or remove memories.

### 4. Agent Reasoning Loop (`jarvis.agent`)

- Executes an iterative action loop where the model produces structured JSON actions (`{"action": "tool", ...}` or `{"action": "final", ...}`).
- Validates tool names and argument schemas before invocation.
- Converts runtime tool outputs or execution errors into feedback messages within a bounded self-repair cycle.
- Automatically extracts durable user facts from turn text and persists them into the memory store.

---

## Hardware Target & Performance Baseline

The system is tested on a local consumer laptop configuration:

- **CPU**: 12th Gen Intel Core i7-12650H (16 logical threads)
- **System Memory**: 38 GB RAM
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (6,144 MB VRAM, Compute Capability 8.6)
- **Host OS**: Ubuntu Linux (x86_64)

### Empirical Highlights

- **KV-Cache Prefix Reuse**: 61.5% reduction in P95 Time-To-First-Token (71.6 ms vs. 186.2 ms) with zero additional VRAM penalty.
- **Hybrid Retrieval Latency**: 11.6 ms (100 memories) to 13.4 ms (1,000 memories) median search latency.
- **Agent Execution**: Autonomous 6-to-7 step task completion with 70% arithmetic precision verified against an independent system oracle.

For detailed empirical benchmark methodologies, tables, and trade-off analyses, refer to [RESULTS.md](file:///run/media/sparsh/CaptainSlow/Programming%20Stuff/Programming%20Stuff/Code%20_n_%20Stuff/All%20Projects/All%20Projects/J.A.R.V.I.S/JARVIS/RESULTS.md).

---

## Installation & Setup

### Requirements

- Linux (Ubuntu 22.04+ recommended)
- Python 3.10+
- NVIDIA GPU with CUDA support and matching drivers

### Environment Setup

```bash
# Clone the repository
git clone <repo_url>
cd JARVIS

# Create and activate conda environment
conda create -n torch_env python=3.10 -y
conda activate torch_env

# Install project dependencies
pip install -r requirements.txt
```

### Model Configuration

By default, JARVIS expects a 4-bit quantized GGUF model:

```bash
export JARVIS_GGUF_PATH="/path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
export JARVIS_N_CTX=8192
export JARVIS_N_GPU_LAYERS=-1
```

---

## Usage

### Interactive CLI

Launch the interactive assistant terminal with real-time token streaming:

```bash
python run_jarvis.py
```

Available CLI commands during interactive mode:
- `/memories`: List all currently indexed memories with metadata.
- `/kv`: Display KV-cache hit and token reuse statistics.
- `/tools`: List available capability tools and schemas.
- `/metrics`: Show generation metrics for the last turn (TTFT, TPS, output tokens).
- `/reset`: Clear memory store and reset KV cache.
- `/quit`: Exit the assistant.

### Automated Demo

Run a non-interactive, multi-turn validation sequence covering memory extraction, calculation, and tool invocation:

```bash
python run_jarvis.py --demo
```

---

## Future Improvements

1. **Constrained Grammar Decoding**: Replace regular-expression JSON extraction with formal context-free grammar constraints (`llama_grammar`) at the C++ decoding level to eliminate JSON syntax errors and malformed tool calls.
2. **Dynamic Context-Aware Retrieval Gating**: Introduce calibrated similarity thresholds to suppress memory injection on casual conversational turns, avoiding distractor retrieval.
3. **Late Interaction / Bi-Encoder Distillation**: Replace pairwise Cross-Encoder reranking with a distilled bi-encoder or ColBERT-style late interaction to achieve high precision without incurring a 100 ms per-query latency penalty.
4. **Embodied Robotics Bridge**: Implement a native ROS 2 node wrapper allowing JARVIS to monitor sensor topics, dispatch waypoint navigation goals, and inspect robot telemetry through capability tools.
5. **Paged KV-Cache Management**: Adopt block-based memory paging across parallel user sessions to prevent memory fragmentation during extended dialogues.
