# Empirical Evaluation and Benchmark Results

This document presents the empirical results of the JARVIS system evaluation across three core architectural dimensions:
1. **Suite A**: KV-Cache and Prompt-Prefix Reuse Latency.
2. **Suite B**: Multi-Step Capability Tool Agent Autonomy and Output Fidelity.
3. **Suite C**: Memory Subsystem Scalability and Cross-Encoder Latency-Quality Trade-Offs.

All benchmarks were executed on physical hardware without mocked components (except where noted) using real quantized model weights, real embedding models, and live Linux telemetry inspection.

---

## 1. System and Hardware Environment

- **Processor**: 12th Gen Intel Core i7-12650H (16 logical cores)
- **Host Memory**: 38,510 MB RAM (~38 GB)
- **Graphics Processing Unit**: NVIDIA GeForce RTX 3060 Laptop GPU (6,144 MB VRAM, Driver Version 595.99.02, Compute Capability 8.6)
- **Operating System**: Linux 7.0.0-42-generic (Ubuntu x86_64)
- **Python Environment**: Python 3.10.21 (`torch_env`)
- **Key Dependencies**:
  - `llama-cpp-python`: 0.3.36
  - `torch`: 2.14.0
  - `transformers`: 5.16.1
  - `sentence-transformers`: 6.0.1
  - `faiss-cpu`: 1.15.0
  - `scikit-learn`: 1.7.2
  - `numpy`: 2.2.6
- **Model Checkpoints**:
  - Primary LLM: `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (4-bit quantized, 4,365 MB VRAM footprint)
  - Memory Embedder: `sentence-transformers/all-MiniLM-L12-v2` (384 dimensions, normalized)
  - Reranker: `BAAI/bge-reranker-base` (Cross-Encoder, sequence length 512)

---

## 2. Suite A: KV-Cache and Cross-Request Prompt-Prefix Reuse

### 2.1 Benchmark Methodology

Suite A evaluates the performance gains from retaining shared prompt prefixes (system instructions and tool schemas) and conversational history in the GPU KV cache across turns. 

- Total Requests: 54 requests (27 under `prefix_reuse_on`, 27 under `prefix_reuse_off_reset_context`).
- Evaluation Sequences: Multi-turn conversations (`growing_history_robotics`) and single-turn queries with identical system prompts (`independent_turns_mixed`).
- Measured Metrics: Time-To-First-Token (TTFT), prefill time, total request time, tokens reused, and peak GPU VRAM.

### 2.2 Results Summary

| Cache Configuration | Requests Evaluated | Cache Hit Rate | Median TTFT (ms) | P95 TTFT (ms) | Median Total Time (ms) | Peak VRAM (MB) |
|:---|---:|---:|---:|---:|---:|---:|
| **Prefix Cache ON** | 27 | **100.0%** (27/27) | **21.07** | **71.61** | **1,173.19** | 4,365.0 |
| **Prefix Cache OFF** | 27 | **0.0%** (0/27) | **29.18** | **186.23** | 1,197.81 | 4,382.0 |

### 2.3 Detailed Findings

1. **Tail Latency Reduction (P95 TTFT)**:
   - When prefix reuse is enabled, the 95th percentile TTFT drops from **186.23 ms to 71.61 ms**, representing a **61.5% reduction** in tail prefill latency.
2. **Median TTFT Speedup**:
   - Median TTFT improves from **29.18 ms to 21.07 ms**, delivering a **1.38x speedup** (27.8% latency reduction).
3. **Token Reuse Volume**:
   - Across the 27 cache-enabled turns, **2,872 prompt tokens** were reused out of **3,674 total prompt tokens**, achieving a **78.17% token-weighted reuse ratio**.
   - Meaningful prefix reuse ($\ge 16$ tokens) was observed on **88.9% (24/27)** of queries. Initial turns reuse the single Beginning-of-Sequence (BOS) token, while all subsequent conversational turns reuse 146+ prefix tokens.
4. **Memory Footprint**:
   - Peak GPU VRAM was **4,365 MB** with caching enabled versus **4,382 MB** with caching disabled. Retaining prefix state in `llama.cpp` incurs zero VRAM penalty while eliminating repeated prompt encoding.

---

## 3. Suite B: Multi-Step Tool Agent Autonomy and Output Fidelity

### 3.1 Benchmark Methodology

Suite B assesses the agent's ability to chain multiple tools, perform arithmetic on raw telemetry outputs, write formatted reports to disk, and provide final natural-language answers.

- **Task**: Inspect GPU VRAM (`nvidia-smi`), system RAM (`free -m`), and CPU activity (`top`/`lscpu`); compute utilization percentages using the `calculate` tool; save a formatted Markdown document to `outputs/system_resource_report.md`; summarize results.
- **Verification**: Outputs are validated against an independent oracle parser that verifies:
  1. Correct extraction of total and used VRAM.
  2. Correct extraction of total, used, and available system RAM.
  3. Strict numerical verification of arithmetic expressions within $\pm 0.5\%$.
  4. Disk artifact existence at the exact relative path, section structure, and SHA-256 integrity.

### 3.2 Results Summary

| Metric | Measured Value | Evaluation Criteria |
|:---|---:|:---|
| Total Runs | 10 | 5 distinct prompt variations $\times$ 2 repetitions |
| GPU Telemetry Inspection Rate | **100.0%** (10/10) | Valid execution and parsing of `nvidia-smi` |
| RAM Telemetry Inspection Rate | **100.0%** (10/10) | Valid execution and parsing of `free -m` |
| Live CPU Inspection Rate | **20.0%** (2/10) | Execution of `top -b -n 1` summary line |
| Arithmetic Calculation Accuracy | **70.0%** (7/10) | Formulas match oracle system utilization $\pm 0.5\%$ |
| Report File Creation & Validity | **50.0%** (5/10) | Valid Markdown report written at required workspace path |
| Mean Tool Steps Taken | **6.5 steps** | Trajectory: inspection $\rightarrow$ calculation $\rightarrow$ file write |
| Median Wall Clock Time | **20,112.59 ms** (~20.1 s) | Complete multi-step loop turnaround time |

### 3.3 Detailed Findings

1. **Tool Chaining Autonomy**:
   - The agent demonstrated multi-turn chaining across tools (`execute_command` $\rightarrow$ `calculate` $\rightarrow$ `filesystem`), completing an average of 6.5 tool operations per task.
2. **Arithmetic Verification**:
   - 70% of calculations performed by the agent matched the exact ground-truth values reported by the operating system kernel within $\pm 0.5\%$.
3. **Failure Analysis & Bottlenecks**:
   - In 8 of 10 runs, the 4B model invoked `lscpu` instead of `top -b -n 1` when given open prompts asking for general "CPU status", obtaining static hardware architecture details rather than dynamic utilization percentages.
   - In 5 runs, the agent produced report files with minor structural omissions (such as writing to the root directory instead of `outputs/` or omitting explicit ISO timestamps).

---

## 4. Suite C: Memory Store Scalability and Cross-Encoder Trade-Off

### 4.1 Benchmark Methodology

Suite C evaluates index build throughput, query latency, and retrieval quality (Recall@K, MRR@5, nDCG@5) across corpus sizes of **100** and **1,000** records.

- **Corpus Characteristics**: 100 hand-curated seed records spanning 5 categories (`goal`, `preference`, `semantic`, `procedural`, `episodic`), expanded to 1,000 records using domain-adjacent distractors.
- **Evaluation Dataset**: 50 curated search queries evaluated against graded relevance judgments (qrels).
- **Configurations Evaluated**:
  - Baseline Hybrid: Dense FAISS (`all-MiniLM-L12-v2`) + Sparse TF-IDF + Reciprocal Rank Fusion (RRF).
  - Hybrid + Cross-Encoder: Baseline candidate union reranked via `BAAI/bge-reranker-base` across batch sizes (8, 16, 32, 64).
  - End-to-End: Hybrid + Cross-Encoder + Maximal Marginal Relevance (MMR) + Query-Aware Router.

### 4.2 Corpus Build Throughput

| Corpus Size | Batch Encode (ms) | FAISS Index Build (ms) | TF-IDF Build (ms) | Total Disk Size | Total Build Time (ms) |
|---:|---:|---:|---:|---:|---:|
| **100** | 46.77 | 0.09 | 3.90 | 189.4 KB | **73.83** |
| **1,000** | 368.51 | 0.19 | 15.85 | 1.95 MB | **396.46** |

Embedding encoding scales linearly ($O(N)$), while FAISS IndexFlatIP building takes under 0.2 ms for 1,000 vectors. Index serialization and verification add under 10 ms.

### 4.3 Query Latency Breakdown (p50 in ms)

| Corpus | Configuration | Batch | Dense Search | Sparse Search | CE Rerank | Total p50 | Total p95 |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 100 | `baseline_no_ce` | - | 0.18 | 0.99 | 0.00 | **11.60** | 13.69 |
| 100 | `hybrid_with_ce_b32` | 32 | 0.29 | 0.97 | 101.49 | **110.41** | 124.95 |
| 100 | `hybrid_ce_batch_8` | 8 | 0.28 | 0.89 | 111.74 | **121.87** | 129.88 |
| 100 | `hybrid_ce_batch_16` | 16 | 0.29 | 1.10 | 100.99 | **112.35** | 128.66 |
| 100 | `hybrid_ce_batch_64` | 64 | 0.30 | 1.12 | 105.48 | **117.81** | 130.07 |
| 1,000 | `baseline_no_ce` | - | 1.07 | 1.73 | 0.00 | **13.37** | 14.64 |
| 1,000 | `hybrid_with_ce_b32` | 32 | 1.07 | 1.85 | 127.94 | **142.36** | 186.72 |
| 1,000 | `hybrid_ce_batch_8` | 8 | 0.78 | 1.44 | 148.44 | **162.32** | 202.64 |
| 1,000 | `hybrid_ce_batch_16` | 16 | 0.78 | 1.49 | 135.42 | **143.08** | 191.04 |
| 1,000 | `hybrid_ce_batch_64` | 64 | 1.09 | 1.85 | 133.43 | **148.68** | 184.13 |
| 1,000 | `end_to_end_ce_mmr_router` | 32 | 1.07 | 1.85 | 130.21 | **144.43** | 186.53 |

*Note: Total p50 includes ~10.2 ms for single-query dense embedding generation via MiniLM.*

### 4.4 Retrieval Relevance Quality

| Corpus | Configuration | Candidate Recall | Recall@1 | Recall@3 | Recall@5 | MRR@5 | nDCG@5 |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 100 | `baseline_no_ce` | **1.0000** | **0.5667** | 0.7000 | **0.8417** | **0.9375** | **0.8829** |
| 100 | `hybrid_with_ce_b32` | **1.0000** | 0.4917 | **0.7250** | 0.8167 | 0.8850 | 0.8318 |
| 1,000 | `baseline_no_ce` | **0.8417** | **0.5667** | 0.6167 | 0.6417 | **0.9250** | **0.8049** |
| 1,000 | `hybrid_with_ce_b32` | **0.8417** | 0.4667 | 0.5917 | 0.6167 | 0.8250 | 0.7228 |
| 1,000 | `end_to_end_ce_mmr_router` | **0.8417** | 0.4667 | **0.6250** | **0.7250** | 0.8333 | 0.7529 |

### 4.5 Cross-Encoder Decision Analysis

| Corpus Size | Candidate Union Size | Baseline p50 | Hybrid+CE p50 | CE Latency Overhead | Overhead % | Quality Delta (MRR@5) |
|---:|---:|---:|---:|---:|---:|---:|
| **100** | 71.2 | **11.60 ms** | 110.41 ms | **+98.81 ms** | **+852%** | -0.0525 |
| **1,000** | 81.8 | **13.37 ms** | 142.36 ms | **+128.99 ms** | **+965%** | -0.1000 |

### 4.6 Empirical Trade-Off Conclusion

1. **Latency Cost**: Cross-Encoder pairwise reranking over 70 to 82 candidates introduces an additional **98 ms to 129 ms per query** (~850% to 965% latency increase), whereas the hybrid FAISS + TF-IDF search completes in **11.6 ms to 13.4 ms**.
2. **Relevance Finding**: The dense + sparse hybrid baseline already achieves a high top-1 Mean Reciprocal Rank (**MRR@5 of 0.9250 to 0.9375**). Pairwise cross-encoder reranking over concise declarative memory statements did not improve top-rank precision and slightly degraded MRR@5 due to high sensitivity to keyword distractors in the expansion set.
3. **Production Recommendation**: For responsive real-time assistant conversations, **`use_cross_encoder=False` is the optimal configuration**, providing a **10x faster retrieval pipeline (13 ms vs. 142 ms)** with superior top-1 relevance.

---

## 5. Limitations and Threats to Validity

1. **Single-GPU Hardware Envelope**: All benchmarks were evaluated on a single 6 GB NVIDIA RTX 3060 Laptop GPU. Concurrent execution of the 4B LLM (~4.4 GB VRAM) alongside the embedding and cross-encoder models (~0.8 GB VRAM) approaches the 6 GB ceiling; evaluations were process-isolated to prevent out-of-memory crashes.
2. **Synthetic Expansion Distractors**: While the 100 seed memories and 50 test queries/qrels were hand-curated across 5 cognitive domains, records 101 to 1,000 were deterministically synthesized domain-adjacent distractors.
3. **Small Model Tool Ambiguity**: In Suite B, the 4B parameter model occasionally chose `lscpu` instead of `top -b -n 1` when given open prompts. Prompt hardening or constrained tool grammars would improve consistency.

---

## 6. Future Improvements

1. **Constrained Grammar Decoding**: Integrate `llama_grammar` support into `LocalLLMEngine` to constrain model generation to valid JSON tool calls, eliminating regex extraction overhead and parsing failures.
2. **Distilled Bi-Encoder Representations**: Fine-tune the MiniLM embedder on domain-specific query-memory pairs using Triplet Margin Loss or Multiple Negatives Ranking Loss to obtain Cross-Encoder-level ranking quality within the fast 13 ms vector search path.
3. **Adaptive Reranker Gating**: Implement conditional reranking where the Cross-Encoder is invoked only when candidate similarity score margins are below a confidence threshold (e.g., top-1 score within 5% of top-2), minimizing latency on confident queries.
4. **Asynchronous Background Memory Maintenance**: Decouple memory persistence, deduplication, and index rebuilding from the interactive query path via background threads to keep user-perceived retrieval latency under 15 ms as the corpus grows beyond 10,000 items.
