# 🤖 J.A.R.V.I.S — Personal AI Assistant & KV-Cache Retrieval Research

A personal AI assistant project combining a modular Python assistant implementation (`jarvis_01.ipynb`) with research and architectural exploration into **Personal KV-Cache Retrieval** and on-device personal agents (`main.ipynb` and `Papers/`).

---

## 📂 Repository Structure

```text
JARVIS/
├── Papers/
│   ├── JUST A RATHER VERY INTELLIGENT SPOKEN AGENT.pdf
│   ├── Jarvis: Towards Personalized AI Assistant via Personal KV-Cache Retrieval.pdf
│   └── OpenJarvis Personal AI On Personal Devices.pdf
├── jarvis_01.ipynb    # Modular J.A.R.V.I.S assistant prototype from scratch
├── main.ipynb         # Study & implementation notes on Personal KV-Cache Retrieval
└── README.md          # Project documentation
```

---

## 🧠 Core Components

### 1. `jarvis_01.ipynb` — Modular Assistant Prototype
Builds a foundational J.A.R.V.I.S engine from scratch in Python before scaling to research-grade LLM architectures:
- **Conversational Output & Formatting**: Structured response interface (`jarvis_speak`) for assistant interaction.
- **System & Environment Telemetry**: Platform inspection, local time/date utilities, file system navigation, and subprocess command execution.
- **Extensible Skill Routing**: Designed to bridge deterministic local utilities with LLM-backed reasoning.

### 2. `main.ipynb` — Personal KV-Cache Retrieval Architecture
Investigates the architecture proposed in *Jarvis: Towards Personalized AI Assistant via Personal KV-Cache Retrieval*:
- **Memory & Identity Layer**: Instead of repeatedly injecting long personal histories into the text prompt, personal evidence is encoded into reusable **Key-Value (KV) cache states**.
- **Selective State Attachment**: Retrieves only the relevant KV-cache blocks at inference time and attaches them directly to the frozen base language model, reducing prefill latency and context bloat.

### 3. `Papers/` — Reference Literature
- **Jarvis: Towards Personalized AI Assistant via Personal KV-Cache Retrieval** — Personal memory representation via KV-cache states.
- **OpenJarvis: Personal AI On Personal Devices** — On-device personal AI assistant systems and local execution workflows.
- **Just A Rather Very Intelligent Spoken Agent** — Spoken dialogue and voice-first agent design.

---

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sparsh0106/JARVIS.git
   cd JARVIS
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook jarvis_01.ipynb
   ```

---

## 👤 Author

**Sparsh Agarwal** — [GitHub](https://github.com/sparsh0106) · [LinkedIn](https://www.linkedin.com/in/sparsh-agarwal-4b443b304/)
