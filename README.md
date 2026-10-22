# :open_book: The Book of AI - Mapping the practical Artificial Intelligence landscape
:toolbox: The Book of Agentic AI — A practical guide to understanding agents and building a practical toolkit you control.

<!-- HOTPLATE:START -->
## :fire: Hotplate (Quarterly Release Cycle)

### :newspaper: General AI vibe for 2026 Q4 {October, November, December} 

:compass: Building and refining open-source agent harnesses for long-running and autonomous work (LOOP for efficiency and duration, OpenHands and OpenCode for coding, goose for general MCP-native use), combined with heavy adoption of MCP tools, persistent memory systems, and self-hosted personal agents that live in messaging apps (OpenClaw and Hermes Agent leading). Strong bias toward tools you can actually run yourself, keep fully under your control, and host on your own hardware or cheap VPS ([Virtual Private Server](https://cloud.google.com/learn/what-is-a-virtual-private-server)).

---


#### Harnesses (the runtime that turns a model into an agent)

- **[Agent Zero](https://www.agent-zero.ai/)** - agentic framework that gives the agent a full Linux computer inside a Docker container.
- **[goose](https://github.com/block/goose)** (Block / Agentic AI Foundation) – Apache-2.0, desktop + CLI + API, heavily MCP-native, local-model friendly.
- **[Hermes Agent](https://github.com/NousResearch/hermes-agent)** (Nous Research) – Python, persistent curated memory, autonomous skill creation, multiple backends (local/Docker/Modal/Daytona). Best long-term accumulation.
- **[OpenClaw](https://github.com/openclaw)**  – TypeScript gateway + 20–30 messaging channels, skills registry (ClawHub / SKILL.md), Docker sandboxes, cron, voice. Best reach.
- **[OpenCode](https://opencode.ai/)** - open source agent that helps you write code in your terminal, IDE, or desktop.
- **[OpenHands](https://github.com/All-Hands-AI/OpenHands)**  – Agent Canvas control center, scoped MCP profiles, isolated Docker runtimes, strong for autonomous coding & repo work.
- **[ZCode](https://zcode.z.ai/en)** - harness and agent for GLM

| Aspect | Hermes Agent | Agent Zero |
| :--- | :--- | :--- |
| **Core Philosophy** | Persistent personal agent that learns & grows over time via skills/memory | Full sandboxed Linux computer + transparent, extensible agent framework |
| **Strengths** | Strong continuous learning loop (auto skill creation/refinement); excellent multi-platform messaging gateway (20+ apps, one memory); flexible deployment (serverless options that hibernate); deep user modeling & cross-session recall; easy always-on personal assistant feel | Real desktop + browser with annotation/cowork; strong project isolation; highly transparent & editable (prompts/tools as files); mature multi-agent hierarchy + Plugin Hub; visual Web UI for control/observation; good for complex computer-use tasks |
| **Weaknesses** | Less emphasis on a full GUI desktop environment; memory/skills can increase token cost on simple tasks; more “gateway + tools” than “own computer” metaphor | Heavier Docker dependency & resource footprint; less native multi-messaging surface out of the box; learning is more manual/project-oriented than automatic skill growth; Web UI-centric rather than chat-app native |
| **Best For** | Always-available personal agent reachable from chat apps that improves with use | Technical users wanting a controllable agentic “computer” for development, research, document work, or multi-agent workflows |
| **Deployment Style** | Lightweight long-running process / serverless; strong messaging focus | Docker-first full environment; Web UI + optional host CLI bridge |
| **Extensibility** | Skills system + MCP + toolsets; community skills | Plugins, skills, fully editable prompts, MCP/A2A, community Plugin Hub |


#### [**MCP**](https://modelcontextprotocol.io/docs/2026-07-28/getting-started/intro)
> is now the default wiring layer (thousands of servers, tool search to keep context clean, permission-aware patterns). Sandboxed runtimes (Docker, E2B, Daytona, Modal, Kernel-style browser infra) and curated persistent memory + agent-written skills are table stakes for anything that runs longer than a single prompt.

#### [KERNEL](https://www.kernel.sh/)
> is a specialized cloud browser infrastructure and "Browser-as-a-Service" platform built specifically for AI web agents and high-scale browser automation. While traditional cloud browsers (like Browserless or standard headless Chrome containers) are designed for traditional web scraping or end-to-end testing, Kernel is engineered around the speed, scale, ***anti-bot bypass***, and execution hurdles inherent to autonomous AI agents (using frameworks like Browser Use, Playwright, Puppeteer, or Stagehand).

#### [HERMES WebUI](https://github.com/nesquena/hermes-webui)
 > is a lightweight, dark-themed web app interface in your browser for Hermes Agent. Full parity with the CLI experience - everything you can do from a terminal, you can do from this UI. No build step, no framework, no bundler. Just Python and vanilla JS.

#### [HONCHO](https://honcho.dev/)
> is open-source AI memory for agents. It remembers each user across sessions, tools and machines, so no session starts cold.
> Pairs well with [Hermes Agent](https://github.com/NousResearch/hermes-agent). [Setup guide](https://honcho.dev/docs/v3/guides/integrations/hermes).

#### [OpenWaldo](https://openwaldo.org/)
| Letter | Meaning |
| --- | --- |
| W | Weights | 
| A | Artifacts |
| L | Licenses |
| D | Data |
| O | Origin |

OpenWALDO brings everything people expect from open source to AI: a community anyone can join; source code and training data anyone can contribute to and audit; open tools; public governance and trust earned through transparency; and models anyone can compose, train, validate, reproduce, extend, and improve.

### [JEV](https://jevmodel.org/)
> A (machine native) decision-focused “System One” model that answers fixed, typed questions with probabilities so your code (or an agent) can act on the result directly, to be consumed by code. It is not a language model. Takes unstructured state + your pre-defined questions and returns reliable, probabilistic decisions that code can immediately branch on.

#### [SpyNEL](https://github.com/agent0ai/spynel)
> Lightweight agent orchestration that uses existing coding harnesses rather than bundling its own AI, with a single communication interface, durable Markdown task management, and autonomous agentic loops.

#### [DOX](https://github.com/agent0ai/dox)
> DOX is a lightweight `AGENTS.md` framework that gives AI agents precise project context through a maintained hierarchy of project-wide and area-specific instructions. Agents read the relevant guidance before editing and update it when meaningful changes affect the documented rules or structure.

#### [Ajax](https://data.pewdiepie.com/)
> Ajax is a fine-tuned (abliterated) Qwen 3.5 9B model, trained for Odysseus to be your always on agent. It handles your daily tasks from search to browse web to email to your calendar - all your daily tasks completely privately. Ajax's refusal has been ablated for a freer, less restricted AI experience. Please use responsibly.

#### [Odysseus](https://odysseusai.dev/)
> Odysseus is a self-hosted AI workspace for chat, agents, deep research, email, calendar, and local/API model backends. It is local-first and privacy-first when you run local models on your own machine.

#### [BenchLM](https://benchlm.ai/)
LLM Leaderboard.
#### [Best Ollama models](https://benchlm.ai/best/ollama-models)

#### [Heretic](https://heretic-project.org/)
Heretic removes restrictions (abliteration parameters) from language models, making sure they always follow your instructions.

### [Capsem](https://docs.capsem.org/getting-started/intro/)
> is a secure runtime / hypervisor for AI agents. It gives each agent its own isolated Linux VM, inspects everything it tries to do on the network or with tools, and logs the activity — so you can run powerful coding agents with much tighter control and visibility than running them directly on your host.
Choose Capsem when your top priority is security and control — you want to run powerful coding agents (or any agent) with strong isolation, network inspection, and auditability.

<!-- HOTPLATE:END -->

--- 

## [Chapter Structure](chatper-structure.md)

--- 


## Introduction

- **An agent** is the autonomous actor — the thing that reasons, plans, uses tools, remembers, and pursues goals over multiple steps.
- **A harness** is the runtime / scaffolding that turns a raw language model into a working agent.
  - _**The harness** is the engine and chassis. **The agent** is the driver + the behavior that emerges once the model is running inside that chassis._


### More precise breakdown

| Concept | What it is | Examples |
| :--- | :--- | :--- |
| **Model** | The raw intelligence (weights) | Qwen, DeepSeek, Claude, Llama, Ajax (abliterated), etc. |
| **Harness** | The runtime that gives the model a loop, tools, memory, sandboxes, skills, messaging, persistence, etc. | Hermes, OpenClaw, goose, OpenHands, OpenCode, Agent Zero |
| **Agent** | The resulting autonomous system that actually does work | Copilot, OpenCode, Agent Zero |

**The harness determines:**
- How long the agent can run
- Whether it has persistent memory and can create its own skills
- What tools (MCP servers) it can reach
- Whether it lives in messaging apps, a terminal, a web UI, or a full Linux desktop
- How isolated / sandboxed its actions are
- How easy it is to observe, interrupt, or improve it

A good harness {Hermes, OPenCode, Agent Zero, OpenClaw, goose, etc.} dramatically changes what kind of agent you can actually have. A weak or missing harness leaves you with just a chatbot that forgets everything between messages


## AI Model Types
- Large Language Models (LLM)
  - General-purpose language models
  - Code-specialized models
    - Claude Fable 5.1 / Claude Opus 5.5 family (Anthropic)
    - GPT-6 Astra / GPT-5.6 Sol / GPT-6.1 Sol series (OpenAI)
    - Gemini 4 Argon / Gemini 3.x Flash family (Google)
    - Kimi K3 / DeepSeek V4.1 Flash / MiMo-V2.6-Pro (open-weight leaders)
    - DeepSeek
    - Qwen
- Image Generation Models
  - Text-to-Image {[Stable Diffusion](https://stablediffusionweb.com/), [Midjourney](https://www.midjourney.com/home), DALL-E, [Grok Imagine](https://grok.com/supergrok/imagine)}
  - Image-to-Image
- Multimodal Models (Text, Image, Audio, Video)
  - Kimi K3 (Moonshot AI)
  - Claude Opus 5 / Claude Opus 5.5 (Anthropic)
  - Qwen3.8 Max (Alibaba)
  - GPT-5.6 series (especially Sol / Luna variants) (OpenAI)
  - Gemini 3.x Flash / Pro family (Google)

### Prominent AI Organizations Creating Foundational AI Models
(Order by Name ↑)

- [Alibaba](https://www.alibabacloud.com/en) (Qwen)
- [Allen Institute for AI](https://allenai.org/)
- [Amazon AI](https://aws.amazon.com/machine-learning/)
- [Anthropic](https://www.anthropic.com/)
- [Black Forest Labs](https://blackforestlabs.ai/) (FLUX)
- [ByteDance](https://www.bytedance.com/) (Seed / Seedream / Doubao)
- [Cohere](https://cohere.ai/)
- [DeepSeek](https://www.deepseek.com/)
- [Google AI / Google DeepMind](https://ai.google/)
- [Hugging Face](https://huggingface.co/)
- [IBM](https://www.ibm.com/granite)
- [Meta AI](https://ai.facebook.com/)
- [Microsoft AI](https://www.microsoft.com/en-us/ai)
- [MiniMax](https://www.minimaxi.com/)
- [Mistral AI](https://mistral.ai/)
- [Moonshot AI](https://www.moonshot.cn/) (Kimi)
- [OpenAI](https://openai.com/)
- [Stability AI](https://stability.ai/)
- [Xiaomi](https://www.xiaomi.com/) (MiMo)
- [xAI](https://x.ai/)
- [Zhipu AI / Z.AI](https://zhipuai.cn/) (GLM)

### Foundational AI Models
(Order by Name ↑)

- [Claude](https://www.anthropic.com/claude) (Anthropic)
- [Command](https://cohere.com/) (Cohere)
- [DeepSeek](https://www.deepseek.com/)
- [FLUX](https://blackforestlabs.ai/) (Black Forest Labs)
- [Gemini](https://ai.google/gemini/) (Google DeepMind)
- [Gemma](https://ai.google.dev/gemma) (Google)
- [GLM](https://zhipuai.cn/) (Zhipu AI / Z.AI)
- [GPT](https://openai.com/) (OpenAI)
- [Granite](https://www.ibm.com/granite) (IBM)
- [Grok](https://x.ai/) (xAI)
- [Kimi](https://www.moonshot.cn/) (Moonshot AI)
- [Llama](https://www.llama.com/) (Meta)
- [MiMo](https://www.xiaomi.com/) (Xiaomi)
- [Mistral](https://mistral.ai/) (Mistral AI)
- [Muse](https://ai.meta.com/) (Meta)
- [Nova](https://aws.amazon.com/ai/generative-ai/nova/) (Amazon)
- [Qwen](https://huggingface.co/Qwen) (Alibaba)
- [Seed / Seedream](https://www.bytedance.com/) (ByteDance)
- [Stable Diffusion](https://stability.ai/stable-image) (Stability AI)

### Base Model Data
(Order by Name ↑)
- [AWS Open Data](https://aws.amazon.com/marketplace/search/results?trk=8384929b-0eb1-4af3-8996-07aa409646bc&sc_channel=el&FULFILLMENT_OPTION_TYPE=DATA_EXCHANGE&CONTRACT_TYPE=OPEN_DATA_LICENSES&filters=FULFILLMENT_OPTION_TYPE%2CCONTRACT_TYPE)
- [**Common Crawl**](https://commoncrawl.org/)
- [**FineWeb**](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) -- curated Internet dataset.
- [**Google Dataset Search**](https://datasetsearch.research.google.com/)
- [**Hugging Face Datasets**](https://huggingface.co/datasets)
- [**Kaggle**](https://www.kaggle.com/datasets)
- [Microsoft Planetary Computer Data Catalog](https://planetarycomputer.microsoft.com/catalog)
- [MNIST (Modified National Institute of Standards and Technology database)](https://github.com/cvdfoundation/mnist)
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
- [SNAP - Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/)
- [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/)
- [USA Data Gov](https://data.gov/)

### Miscellaneous Tools
(Order by Name ↑)
- [Excalidraw](https://excalidraw.com/) - online Diagram tool
- [LLM Visualization](https://bbycroft.net/llm)
- [Tiktokenizer](https://tiktokenizer.vercel.app/)

### Benchmarks
("Trust me bro benchmarks")
| Benchmark | Status in 2026 | Strengths | Weaknesses | Still useful? |
|-----------|----------------|-----------|------------|---------------|
| **MMLU** | Saturated | Historical reference for broad knowledge | Frontier models cluster ~90%+; little differentiation; contamination risk | Mostly no (use MMLU-Pro if needed) |
| **GSM8K** | Saturated | Simple math word problems | Top models near ceiling (~95%+); many invalid questions noted | No for frontier comparison |
| **GPQA** (Graduate-Level Google-Proof Q&A Benchmark; esp. Diamond) | Near-saturating | Challenging multiple-choice benchmark evaluating advanced scientific reasoning and deep domain understanding in large language models | Leaders pushing 90%+; small set → noise | Still decent for science reasoning, but headroom shrinking |
| **SWE-bench** (Verified/Pro variants) | Contested / partially saturating | Real GitHub issue resolution; strong signal for coding agents | Contamination concerns; scores climbing into high 70s–90s on easier versions | Yes for coding/agentic work (prefer newer/harder variants like Pro) |
| **GAIA** | Active but contested | Multi-step agentic tasks (browsing, tools, files) that are easy for humans | Large score gaps depending on scaffolding/tools; some saturation reports | Yes for assistant/agent evaluation |
| [**LiveBench**](https://livebench.ai/#/) | Highly regarded | Monthly-refreshed questions from recent sources; objective ground-truth scoring; covers reasoning, coding, math, data analysis, language, instruction following | Not human preference; not pure agentic long-horizon | **One of the strongest current objective capability benchmarks** |
| [**LMArena**](https://lmarena.ai/) (Chatbot/Arena) | Highly regarded | Large-scale blind human preference (Elo-style); real user prompts | Style/verbosity bias; overlapping confidence intervals; not verifiable correctness | **Best for “which model feels best in open-ended use”** |
| [**BenchLM**](https://benchlm.ai/) | Aggregator | Combines many sources into overall indices | Depends on underlying benchmarks; not a primary eval itself | Useful meta-view, not a standalone benchmark |
| **Mensa Norway IQ Test** | Niche | Attempts IQ-style scoring | Narrow, not standard in frontier AI evaluation | Rarely used for serious model comparison |
| **WebDev Arena** | Specialized | Web development tasks | Narrow domain | Useful only if you care specifically about web-dev agents |


### Running AI Models Locally

- [Ollama](https://ollama.com) — CLI + local OpenAI-compatible API (most popular for developers)
- [LM Studio](https://lmstudio.ai/) — Polished desktop GUI
- [Jan](https://jan.ai/) — Fully open-source desktop app (privacy-focused)
- [Open WebUI](https://openwebui.com/) — Self-hosted web UI (excellent with Ollama)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Core inference engine (GGUF, maximum control)
- [KoboldCpp](https://github.com/LostRuins/koboldcpp) — Easy single-binary runner with built-in UI
- [vLLM](https://github.com/vllm-project/vllm) — High-throughput serving (multi-user / production)
- [jan](https://www.jan.ai/) - run models locally, with cloud integration, MCP support.

| Aspect              | **LM Studio**                                      | **Ollama**                                          |
|---------------------|----------------------------------------------------|-----------------------------------------------------|
| **Primary focus**   | Polished desktop GUI for discovery, chat & experimentation | CLI + background service for scripting, APIs & integrations |
| **License**         | Proprietary (free for personal + internal business use) | Fully open-source (MIT)                             |
| **Interface**       | Excellent GUI + CLI (`lms`) + headless daemon (`llmster`) | CLI-first + local API; has added a basic desktop app |
| **Model discovery** | Built-in Hugging Face browser with size/VRAM estimates | Curated library + simple `ollama pull model:tag`    |
| **API**             | OpenAI- + Anthropic-compatible (port 1234)         | OpenAI-compatible + native endpoints (port 11434)   |
| **Best hardware edge** | Stronger MLX performance on Apple Silicon         | Lower overhead, often slightly faster on NVIDIA    |
| **Headless/server** | Yes (llmster since early 2026)                     | Native from the start (systemd/Docker-friendly)     |
| **Extras**          | Built-in document chat (RAG), MCP client, LM Link (remote device sharing), continuous batching | Excellent ecosystem integrations, Modelfiles, official Docker image, optional cloud models |


#### When Ollama is usually better: You want a fully open-source, auditable tool.
- You are a developer building scripts, agents, or apps (many frameworks and coding tools assume Ollama by default).
- You need Docker, Kubernetes, or long-running headless servers with minimal footprint.
- You prefer the lowest idle RAM and fastest cold-start times (especially on NVIDIA GPUs).
- You want the absolute simplest CLI workflow (ollama run model).


## Web/Online Models
(Order by Name ↑)

- [Anthropic](https://www.anthropic.com/) - [Claude](https://claude.ai/)
- [DeepSeek](https://www.deepseek.com/) - [Chat](https://chat.deepseek.com/)
- [GitHub Models](https://github.com/marketplace/models)
- [Google](https://ai.google/) - [Gemini](https://gemini.google.com/) / [AI Studio](https://aistudio.google.com/)
- [Grok](https://grok.com/) (xAI)
- [IBM Granite](https://www.ibm.com/granite/playground/)
- [Kimi](https://kimi.moonshot.cn/) (Moonshot AI)
- [Mammouth](https://mammouth.ai/) - Multi-model selector
- [Meta AI](https://www.meta.ai/)
- [Microsoft Copilot](https://copilot.microsoft.com/)
- [Mistral](https://chat.mistral.ai/)
- [OpenAI](https://openai.com/) - [ChatGPT](https://chatgpt.com/)
- [Perplexity](https://www.perplexity.ai/)
- [Qwen](https://chat.qwen.ai/) (Alibaba)
- [Together AI](https://www.together.ai/)
- [TypeSafe](https://console.typesafe.ai/home)
- [Z](https://chat.z.ai/) (GLM)


## AI [blog] Resources
prioritise independent writers, technical depth, open-model communities, and low-hype signal over mainstream media.

(Order by Name ↑)

- [AI News](https://news.smol.ai/)
- [AI News Dashboard](https://news.kabyik.dev/)
- [Ai2 – Allen Institute](https://allenai.org/blog)
- [ApX Machine Learning](https://apxml.com/posts)
- [Arsturn](https://www.arsturn.com/blog)
- [Epoch AI](https://epoch.ai/)
- [Hugging Face](https://huggingface.co/blog)
- [Import AI](https://importai.substack.com/) (Jack Clark)
- [Interconnects](https://www.interconnects.ai/) (Nathan Lambert)
- [Latent Space](https://www.latent.space/)
- [SemiAnalysis](https://www.semianalysis.com/)
- [ShinChven's Blog](https://atlassc.net/)
- [Simon Willison’s Weblog](https://simonwillison.net/)
- [The Batch](https://www.deeplearning.ai/the-batch/) (Andrew Ng)
- [The Gradient](https://thegradient.pub/)
- [Turning Post](https://www.turingpost.com/)


## Prominent AI People
(Sorted by Surname ↑)

- [Diogo Almeida](https://typesafe.ai/team) - [TypeSafe](https://typesafe.ai/)
- [Dario Amodei](https://darioamodei.com/) - [Anthropic](https://www.anthropic.com/)
- [Sam Altman](https://openai.com/about/) - [OpenAI](https://openai.com/)
- [AmandA Askell](https://askell.io/) - [Anthropic](https://www.anthropic.com/)
- [Yoshua Bengio](https://yoshuabengio.org/en/) - [Mila / Université de Montréal](https://mila.quebec/)
- [Lex Fridman](https://lexfridman.com/) - [MIT](https://www.mit.edu/)
- [Demis Hassabis](https://deepmind.google/about/) - [Google DeepMind](https://deepmind.google/)
- [Geoffrey Hinton](https://discover.research.utoronto.ca/26059-geoffrey-e-hinton) - [University of Toronto](https://www.utoronto.ca/)
- [Jensen Huang](https://www.nvidia.com/en-us/about-nvidia/board-of-directors/jensen-huang/) - [NVIDIA](https://www.nvidia.com/)
- [Salim Ismail](https://salimismail.com/) - [Anthropic](https://www.anthropic.com/)
- [Nathan Lambert](https://www.interconnects.ai/) - [Allen Institute for AI](https://allenai.org/)
- [Yann LeCun](http://yann.lecun.com/) - [Meta / NYU](https://www.nyu.edu/)
- [Fei-Fei Li](https://drfeifei.substack.com/about) - [Stanford University / World Labs](https://www.stanford.edu/)
- [Emad Mostaque](https://emad.posthaven.com/) - [stability.ai](https://stability.ai/)
- [Andrew Ng](https://www.andrewng.org/) - [DeepLearning.AI / Stanford](https://www.deeplearning.ai/)
- [Christopher Olah](https://colah.github.io/) - [Anthropic](https://www.anthropic.com/)
- [Ilya Sutskever](https://ssi.inc/) - [Safe Superintelligence (SSI)](https://ssi.inc/)


## AI Publication Websites
(Order by Name ↑)

- [arxiv](https://arxiv.org/list/cs.AI/recent)


## AI Publications
(Sorted by Publication Date ↓)

- 2025-04-25 [BitNet b1.58 2B4T Technical Report](https://arxiv.org/abs/2504.12285)
- 2024-04-09 [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654)
- 2023-07-27 [Universal and Transferable Adversarial Attacks on Aligned Language Models - LLM Attacks](https://arxiv.org/abs/2307.15043)
- 2020-05-28 [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- 2019-06-17 [Superposition of many models into one](https://arxiv.org/abs/1902.05563)
- 2018-10-11 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- 2017-06-12 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 2015-12-10 [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
- 2014-06-10 [Generative Adversarial Networks (GANs)](https://arxiv.org/abs/1406.2661)
- 2012 [ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)](https://dl.acm.org/doi/10.1145/3065386)

## Articles
(Sorted by Publication Date ↓)

- 2026-09-11 [Lies, Damned Lies, and Benchmarks](https://www.completeskeptic.com/p/lies-damned-lies-and-benchmarks)
- 2026-09-09 [(KV) Cache Rules Everything Around Me](https://www.completeskeptic.com/p/kv-cache-rules-everything-around)
- 2026-08-27 [Pivot to AI: Your AGENTS.md file doesn’t actually do anything](https://pivot-to-ai.com/2026/08/27/your-agents-md-file-doesnt-actually-do-anything/)
- 2025-02-12 [Unlocking the Effective Context Length: Benchmarking the Granite-3.1-8b Model](https://www.redhat.com/en/blog/unlocking-effective-context-length-benchmarking-granite-31-8b-model)
- 2025-02-02 [A Detailed Analysis of Fine-Tuning, Direct Preference Optimization (DPO), and Reinforcement Learning with Verifiable Rewards (RLVR) on the LLama3.1 405B Model](https://medium.com/@zhouboyang1983/a-detailed-analysis-of-fine-tuning-direct-preference-optimization-dpo-and-reinforcement-c24d9061cd84)
- 2025-01-31 [DeepSeek-V3 Explained 1: Multi-head Latent Attention](https://towardsdatascience.com/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4/)
- 2025-01-19  [Top 5 Mistakes to Avoid When Learning Machine Learning](https://apxml.com/posts/top-mistakes-when-learning-machine-learning)
- 2024-10-02  [How to Get Started with Machine Learning: A Beginner’s Step-by-Step Guide](https://apxml.com/posts/get-started-with-machine-learning-guide)
- 2024-07-13  [MHA vs MQA vs GQA vs MLA](https://medium.com/@zaiinn440/mha-vs-mqa-vs-gqa-vs-mla-c6cf8285bbec)
- 2020-00-00  [Over 200 of the Best Machine Learning, NLP, and Python Tutorials — 2018 Edition](https://robbieallen.medium.com/over-200-of-the-best-machine-learning-nlp-and-python-tutorials-2018-edition-dd8cf53cb7dc)

## Online Books
(Sorted by Publication Date ↓)

- 2026  [Principles and Practice of Deep Representation Learning](https://ma-lab-berkeley.github.io/deep-representation-learning-book/)
- 2024  [Machine Learning and Deep Learning with R](https://theoreticalecology.github.io/machinelearning/)
- 2024  [Natural Language Processing (NLP) Course](https://srdas.github.io/NLPBook/intro.html)
- 2023  [Dive into Deep Learning](https://d2l.ai/)
- 2023  [State of Open Source AI Book - 2023 Edition](https://book.premai.io/state-of-open-source-ai/)
- 2023  [Understanding Deep Learning](https://udlbook.github.io/udlbook/)
- 2019  [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- 2018  [Deep Learning](https://srdas.github.io/DLBook/)
- 2016  [Deep Learning](https://www.deeplearningbook.org/)

## YouTube Videos & Channels
(Sorted by Publication Date ↓)
- 2026-10-21  [Kai: I Tested Every Best Local AI For GPU (8GB TO 24GB VRAM)](https://youtu.be/ugKAkCVBcz0?si=aRJ9Rx0guKR-Bls6)
- 2026-10-21  [Kai: Your Local AI Is Not as Private as You Think](https://youtu.be/3mx84MRj8lY?si=ffjk7oazRh3vbYrr)
- 2026-10-19  [Cloud Codes: Your Local AI Model Isn’t as Safe as You Think](https://youtu.be/7S3kCU1tXcA?si=ABvWS3BS3Og4Va72)
- 2026-10-11  [Stanford CS146S: How always-on agents work](https://youtu.be/L1OBRtfPS34?si=oejX5mZMiLUkk0rR)
- 2026-09-21  [Why I couldn't build Jev at OpenAI — Diogo Almeida, TypeSafe Co-founder & CEO](https://youtu.be/cFx9Z3ZXca0?si=qHXpijFYS21TGtHX)
- 2026-05-13  [The Coding Gopher: Uni-1 AI image generation model](https://youtu.be/xhaE2sUnUZE?si=otuy7ytKM3nXzbFn) 
- 2026-03-26  [The Diary of a CEO: AI Whistleblower: We Are Being Gaslit By AI Companies, They’re Hiding The Truth! - Karen Hao](https://youtu.be/Cn8HBj8QAbk?si=1CGH8ELHuUwIP39-)
- 2025-04-28  [Logically Answered: AMD's $243 Billion AI Disaster...What Happened?](https://youtu.be/SDIq26_DVLk?si=RqlxfKt6Gff-IG4M)
- 2025-04022  [Computerphile: What is Cuda?](https://youtu.be/K9anz4aB0S0?si=VO5Uy9iJ5AKajEnP)
- 2025-03-05  [ByteByteGo: What Is the Most Popular Open-Source AI Stack?](https://www.youtube.com/watch?v=hFURlsMwU7c)
- 2025-02-05  [Andrej Karpathy: Deep Dive into LLMs like ChatGPT](https://youtu.be/7xTGNNLPyMI?si=u2q8zaBubzTCPKC7)
- 2025-02-02  [Lex Fridman: DeepSeek, China, OpenAI, NVIDIA, xAI, TSMC, Stargate, and AI Megaclusters | Lex Fridman Podcast #459](https://youtu.be/_1f-o0nqpEI?si=b3xO1D6jP-5g06e8)
- 2025-01-29  [Peter H. Diamandis: DeepSeek vs. Open AI - The State of AI w/ Emad Mostaque & Salim Ismail | EP #146](https://youtu.be/lY8Ja00PCQM?si=71XmR5B_VikMcYCg)
- 2024-11-11  [Lex Fridman: Dario Amodei: Anthropic CEO on Claude, AGI & the Future of AI & Humanity | Lex Fridman Podcast #452](https://youtu.be/ugvHCXCOmm4?si=QekDqk5yuNA5fR1H)
- 2024-09-09  [IBM Technology: RAG vs. Fine Tuning](https://youtu.be/00Q0G84kq3M?si=iJn3KAoFzOMM8a-L)
- 2024-04-01  [3Blue1Brown: Neural networks Playlist](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=xFClf3TeCbAQkgok)
- 2023-08-23  [IBM Technology: What is Retrieval-Augmented Generation (RAG)?](https://youtu.be/T-D1OfcDW1M?si=Uv-M2QcZL7WIJi40)
- 2018-07-30  [Lex Fridman: Deep Learning State of the Art (2020)](https://youtu.be/0VH1Lim8gL8?si=84lMRmcAPNYN6Lm-)
- [Channel: StatQuest with Josh Starmer](https://www.youtube.com/@statquest)

## Quotes
> "Not all benchmarks lie, but all liars benchmark"

## [Glossory](Glossory.md)
## [Notes](Notes.md)

## History
_This is a work in progress --started 2025-02-- to document [my](https://github.com/DavidGeeraerts) exploration of Artificial Intelligence, Large Language Models (LLM), Natural Language Models (NLM), Natural Language Processing (NLP), etc._



## License
Unless otherwise noted, original content in this repository by David Geeraerts is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE). Please attribute it as “The Book of AI by David Geeraerts” and link to this repository and the license. Third-party materials, including research papers and images, are excluded and remain subject to their own terms.

## 📊 GitHub Stats
![Alt](https://repobeats.axiom.co/api/embed/a45c742cf7979dcff3ac7e99445bc37a05abb2d0.svg "Repobeats analytics image")