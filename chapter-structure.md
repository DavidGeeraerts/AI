# Chapter Structure

## 1. Introduction
**Goal**: Orient the reader and set the practical, control-oriented tone of the book.

- Purpose of the book and the repository  
- The 2026 Q4 “Hotplate” reality check  
  - Dominant themes: long-running open-source agent harnesses, MCP as default wiring, persistent memory, self-hosted personal agents living in messaging apps  
  - Strong bias toward tools you can run yourself, keep under your control, and host on your own hardware or cheap VPS  
- How the quarterly Hotplate system works  
- Who this book is for (practicing developers, indie builders, teams that want local-first stacks)  
- How to use the book as a living toolkit rather than a static textbook  
- Quick map of the rest of the chapters  

## 2. Models
**Goal**: Give the reader a clear mental model of the current model landscape so they can choose intelligently.

- Model categories  
  - Large Language Models (general-purpose and code-specialized)  
  - Multimodal models  
  - Image generation models (text-to-image, image-to-image)  
  - Decision / “System One” models (Jev and the calibrated-decision paradigm)  

- Major organizations and foundational model families  
  - Anthropic (Claude family)  
  - OpenAI (GPT series)  
  - Google DeepMind (Gemini / Gemma)  
  - DeepSeek, Qwen (Alibaba), Meta (Llama / Muse), xAI (Grok), Moonshot (Kimi), Mistral, Zhipu (GLM), etc.  

- Open-weight vs closed models and the practical implications  
- Abliteration / refusal removal  
  - Why it matters for agent autonomy  
  - Key tools: Heretic, Ajax (and related abliterated fine-tunes)  

- Benchmarks that still matter in 2026  
  - Saturated ones (MMLU, GSM8K) vs still-useful ones (SWE-bench variants, LiveBench, LMArena, GAIA, BenchLM)  
  - How to read leaderboards intelligently  

- Training data sources (high-level overview of Common Crawl, FineWeb, etc.)  

## 3. Running a Model
**Goal**: Get the reader from zero to a working local (and hybrid) inference setup.

### 3.1 Local-First Inference
- Core runners and their roles  
  - **Ollama** — CLI + local OpenAI-compatible API (developer default)  
  - **llama.cpp** — maximum control, GGUF, low-level engine  
  - **LM Studio** — polished desktop GUI + API  
  - **Jan** — fully open-source desktop app with emerging MCP support  
  - **Open WebUI** — self-hosted web frontend  
  - **KoboldCpp**, **vLLM** (high-throughput serving)  

- Practical comparison tables (Ollama vs LM Studio, hardware considerations, Apple Silicon vs NVIDIA, RAM/VRAM requirements)  
- Model management: pulling, quantizing, Modelfiles, custom system prompts  
- Exposing models as OpenAI-compatible endpoints for agents and tools  

### 3.2 Cloud and Hybrid Options
- Major hosted providers and when they still make sense  
- Hybrid patterns: local for privacy/cost/latency, cloud for peak capability  
- Cost, rate-limit, and data-sovereignty trade-offs  

### 3.3 Making Models Usable by Agents
- Serving considerations (context length, tool-calling support, structured output)  
- Configuration templates (drawing from the repo’s `configuration/` directory)  

## 4. Agents and Harnesses
**Goal**: Explain the runtime layer that turns a model into something that can do useful, long-running work.

- What a harness is (and why “just calling the model” is no longer enough)  
- Core modern harnesses (Hotplate focus)  
  - **Hermes Agent** (Nous Research) — persistent curated memory, autonomous skill creation, multi-backend  
  - **OpenClaw** — TypeScript gateway, 20–30 messaging channels, skills registry, Docker sandboxes  
  - **OpenHands** — Agent Canvas, scoped MCP profiles, strong for autonomous coding & repo work  
  - **OpenCode** — terminal / IDE / desktop coding agent  
  - **goose** (Block) — MCP-native, desktop + CLI + API, local-model friendly  
  - **Agent Zero** — full Linux computer inside Docker, transparent multi-agent hierarchy  

- Comparison matrix (philosophy, strengths, weaknesses, best-fit use cases, deployment style)  

- Supporting infrastructure that has become table stakes  
  - **MCP** (Model Context Protocol) as the default tool-wiring layer  
  - Sandboxed execution environments (Docker, Daytona, Modal, E2B, Kernel browser infrastructure)  
  - Persistent memory systems and agent-written skills  
  - **SpyNEL** — lightweight orchestration that sits on top of existing coding harnesses  
  - **DOX / AGENTS.md** hierarchy — giving agents precise, maintainable project context  

## 5. Building a Working Stack
**Goal**: Move from individual components to a coherent, usable system.

- Minimal viable personal agent stack  
  - Local model (Ollama / Jan) + chosen harness + MCP tools + memory  
- Always-on personal agents  
  - Odysseus workspace, Ajax, messaging-gateway patterns (OpenClaw + Hermes)  
- Coding-focused stacks  
  - OpenHands / OpenCode + repo context (DOX) + sandboxed execution  
- Incorporating decision models  
  - Using Jev-style calibrated decisions (RLCD) inside agent loops for routing, gating, and confidence-based escalation  
- Practical concerns  
  - Containment and permission models  
  - Cost and token economics of long-running agents  
  - Observability and debugging multi-step agent runs  
  - Updating skills and memory without breaking the system  

## 6. Miscellaneous Tools & Supporting Infrastructure
**Goal**: Cover the useful secondary tools that make daily work smoother.

- Understanding and visualization  
  - Tiktokenizer, LLM Visualization, Excalidraw  
- Model selection and comparison  
  - BenchLM, Best Ollama models lists  
- Abliteration and model modification  
  - Heretic and related tooling  
- Open-source principles for AI  
  - OpenWALDO (Weights, Artifacts, Licenses, Data, Origin)  
- Signal sources worth following  
  - Independent blogs and technical writing prioritized by the repo (Interconnects, Latent Space, Simon Willison, SemiAnalysis, etc.)  
  - Key papers and historical milestones listed in the repo  

## 7. Glossary & Reference
**Goal**: Provide a quick, practitioner-focused reference.

- Condensed, high-signal entries drawn from the repo’s Glossory.md  
  - Abliteration, Agentic AI, Alignment, Attention, Benchmarks, BPE, MCP, RLHF, RLVR, RLCD, GRPO, System One models, etc.  
- Cross-references back to the chapters where the concepts are used in practice  

## 8. Keeping the Map Alive (Appendix / Closing)
**Goal**: Make the book a living resource rather than a snapshot that ages quickly.

- How the Hotplate is maintained and released quarterly  
- Suggested process for keeping your own local copy current  
- How to contribute observations or corrections  
- Recommended next steps by persona  
  - Solo developer / indie hacker  
  - Team lead introducing agentic workflows  
  - Researcher or evaluator  
- Final encouragement: the highest-leverage skill is the ability to assemble and control your own stack  

---
