# 📖 The Book of AI - Mapping the practical Artificial Intelligence landscape

🧰 The Book of Agentic AI — A practical guide to understanding agents and building a practical toolkit you control.

## 🔥 Hotplate (Quarterly Release Cycle)

## 🔥 Hotplate (Quarterly Release Cycle)

### 📰 General AI vibe for 2026 Q4 {October, November, December}

🧭 Building and refining open-source agent harnesses for long-running and autonomous work (LOOP for efficiency and duration, OpenHands and OpenCode for coding, goose for general MCP-native use), combined with heavy adoption of MCP tools, persistent memory systems, and self-hosted personal agents that live in messaging apps (OpenClaw and Hermes Agent leading). Strong bias toward tools you can actually run yourself, keep fully under your control, and host on your own hardware or cheap VPS ([Virtual Private Server](https://cloud.google.com/learn/what-is-a-virtual-private-server)).

------------------------------------------------------------------------

#### Harnesses (the runtime that turns a model into an agent)

- **[Agent Zero](https://www.agent-zero.ai/)** - agentic framework that gives the agent a full Linux computer inside a Docker container.
- **[goose](https://github.com/block/goose)** (Block / Agentic AI Foundation) – Apache-2.0, desktop + CLI + API, heavily MCP-native, local-model friendly.
- **[Hermes Agent](https://github.com/NousResearch/hermes-agent)** (Nous Research) – Python, persistent curated memory, autonomous skill creation, multiple backends (local/Docker/Modal/Daytona). Best long-term accumulation.
- **[OpenClaw](https://github.com/openclaw)** – TypeScript gateway + 20–30 messaging channels, skills registry (ClawHub / SKILL.md), Docker sandboxes, cron, voice. Best reach.
- **[OpenCode](https://opencode.ai/)** - open source agent that helps you write code in your terminal, IDE, or desktop.
- **[OpenHands](https://github.com/All-Hands-AI/OpenHands)** – Agent Canvas control center, scoped MCP profiles, isolated Docker runtimes, strong for autonomous coding & repo work.
- **[ZCode](https://zcode.z.ai/en)** - harness and agent for GLM

| Aspect | Hermes Agent | Agent Zero |
|:---|:---|:---|
| **Core Philosophy** | Persistent personal agent that learns & grows over time via skills/memory | Full sandboxed Linux computer + transparent, extensible agent framework |
| **Strengths** | Strong continuous learning loop (auto skill creation/refinement); excellent multi-platform messaging gateway (20+ apps, one memory); flexible deployment (serverless options that hibernate); deep user modeling & cross-session recall; easy always-on personal assistant feel | Real desktop + browser with annotation/cowork; strong project isolation; highly transparent & editable (prompts/tools as files); mature multi-agent hierarchy + Plugin Hub; visual Web UI for control/observation; good for complex computer-use tasks |
| **Weaknesses** | Less emphasis on a full GUI desktop environment; memory/skills can increase token cost on simple tasks; more “gateway + tools” than “own computer” metaphor | Heavier Docker dependency & resource footprint; less native multi-messaging surface out of the box; learning is more manual/project-oriented than automatic skill growth; Web UI-centric rather than chat-app native |
| **Best For** | Always-available personal agent reachable from chat apps that improves with use | Technical users wanting a controllable agentic “computer” for development, research, document work, or multi-agent workflows |
| **Deployment Style** | Lightweight long-running process / serverless; strong messaging focus | Docker-first full environment; Web UI + optional host CLI bridge |
| **Extensibility** | Skills system + MCP + toolsets; community skills | Plugins, skills, fully editable prompts, MCP/A2A, community Plugin Hub |

#### [**MCP**](https://modelcontextprotocol.io/docs/2026-07-28/getting-started/intro)

is now the default wiring layer (thousands of servers, tool search to keep context clean, permission-aware patterns). Sandboxed runtimes (Docker, E2B, Daytona, Modal, Kernel-style browser infra) and curated persistent memory + agent-written skills are table stakes for anything that runs longer than a single prompt.

#### [KERNEL](https://www.kernel.sh/)

is a specialized cloud browser infrastructure and “Browser-as-a-Service” platform built specifically for AI web agents and high-scale browser automation. While traditional cloud browsers (like Browserless or standard headless Chrome containers) are designed for traditional web scraping or end-to-end testing, Kernel is engineered around the speed, scale, ***anti-bot bypass***, and execution hurdles inherent to autonomous AI agents (using frameworks like Browser Use, Playwright, Puppeteer, or Stagehand).

#### [OpenWaldo](https://openwaldo.org/)

| Letter | Meaning   |
|--------|-----------|
| W      | Weights   |
| A      | Artifacts |
| L      | Licenses  |
| D      | Data      |
| O      | Origin    |

OpenWALDO brings everything people expect from open source to AI: a community anyone can join; source code and training data anyone can contribute to and audit; open tools; public governance and trust earned through transparency; and models anyone can compose, train, validate, reproduce, extend, and improve.

### [JEV](https://jevmodel.org/)

A (machine native) decision-focused “System One” model that answers fixed, typed questions with probabilities so your code (or an agent) can act on the result directly, to be consumed by code. It is not a language model. Takes unstructured state + your pre-defined questions and returns reliable, probabilistic decisions that code can immediately branch on.

#### [SpyNEL](https://github.com/agent0ai/spynel)

Lightweight agent orchestration that uses existing coding harnesses rather than bundling its own AI, with a single communication interface, durable Markdown task management, and autonomous agentic loops.

#### [DOX](https://github.com/agent0ai/dox)

DOX is a lightweight `AGENTS.md` framework that gives AI agents precise project context through a maintained hierarchy of project-wide and area-specific instructions. Agents read the relevant guidance before editing and update it when meaningful changes affect the documented rules or structure.

#### [Ajax](https://data.pewdiepie.com/)

Ajax is a fine-tuned (abliterated) Qwen 3.5 9B model, trained for Odysseus to be your always on agent. It handles your daily tasks from search to browse web to email to your calendar - all your daily tasks completely privately. Ajax’s refusal has been ablated for a freer, less restricted AI experience. Please use responsibly.

#### [Odysseus](https://odysseusai.dev/)

Odysseus is a self-hosted AI workspace for chat, agents, deep research, email, calendar, and local/API model backends. It is local-first and privacy-first when you run local models on your own machine.

#### [BenchLM](https://benchlm.ai/)

LLM Leaderboard. \#### [Best Ollama models](https://benchlm.ai/best/ollama-models)

#### [Heretic](https://heretic-project.org/)

Heretic removes restrictions (abliteration parameters) from language models, making sure they always follow your instructions.

### [Capsem](https://docs.capsem.org/getting-started/intro/)

is a secure runtime / hypervisor for AI agents. It gives each agent its own isolated Linux VM, inspects everything it tries to do on the network or with tools, and logs the activity — so you can run powerful coding agents with much tighter control and visibility than running them directly on your host. Choose Capsem when your top priority is security and control — you want to run powerful coding agents (or any agent) with strong isolation, network inspection, and auditability.

## Index

[Chapter-01-Introduction](chapters/01-introduction.llms.md) provides an overview of the book’s purpose and structure.

[Chapter-02-Models](chapters/02-models.llms.md) covers the fundamental models in AI.

[Chapter-03-Running-a-Model](chapters/03-running-a-model.llms.md) explains how to run and use these models.

[Chapter-04-Agents-and-Harnesses](chapters/04-agents-and-harnesses.llms.md) discusses the role of agents and harnesses in AI.

[Chapter-05-Building-a-Working-Stack](chapters/05-building-a-working-stack.llms.md) guides you through building a functional AI stack.

[Chapter-06-Miscellaneous-Tools](chapters/06-miscellaneous-tools.llms.md) lists various tools that are useful in AI development.

[Chapter-07-Resources](chapters/07-resources.qmd) provides a collection of resources for further learning.

[Chapter-08-Keeping-the-Map-Alive](chapters/08-keeping-the-map-alive.llms.md) offers tips for maintaining and updating your AI knowledge base.

[Appendix-Glossary](appendices/Glossary.llms.md) contains definitions of key terms used throughout the book.

[Appendix-Notes](appendices/Notes.llms.md) includes additional notes and references for deeper understanding.

Back to top
