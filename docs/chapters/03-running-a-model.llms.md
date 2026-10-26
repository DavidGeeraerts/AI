# Running a Model

Published

2026-10-26

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

### Running AI Models Locally

- [Ollama](https://ollama.com) — CLI + local OpenAI-compatible API (most popular for developers)
- [LM Studio](https://lmstudio.ai/) — Polished desktop GUI
- [Jan](https://jan.ai/) — Fully open-source desktop app (privacy-focused)
- [Open WebUI](https://openwebui.com/) — Self-hosted web UI (excellent with Ollama)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Core inference engine (GGUF, maximum control)
- [KoboldCpp](https://github.com/LostRuins/koboldcpp) — Easy single-binary runner with built-in UI
- [vLLM](https://github.com/vllm-project/vllm) — High-throughput serving (multi-user / production)
- [jan](https://www.jan.ai/) - run models locally, with cloud integration, MCP support.

| Aspect | **LM Studio** | **Ollama** |
|----|----|----|
| **Primary focus** | Polished desktop GUI for discovery, chat & experimentation | CLI + background service for scripting, APIs & integrations |
| **License** | Proprietary (free for personal + internal business use) | Fully open-source (MIT) |
| **Interface** | Excellent GUI + CLI (`lms`) + headless daemon (`llmster`) | CLI-first + local API; has added a basic desktop app |
| **Model discovery** | Built-in Hugging Face browser with size/VRAM estimates | Curated library + simple `ollama pull model:tag` |
| **API** | OpenAI- + Anthropic-compatible (port 1234) | OpenAI-compatible + native endpoints (port 11434) |
| **Best hardware edge** | Stronger MLX performance on Apple Silicon | Lower overhead, often slightly faster on NVIDIA |
| **Headless/server** | Yes (llmster since early 2026) | Native from the start (systemd/Docker-friendly) |
| **Extras** | Built-in document chat (RAG), MCP client, LM Link (remote device sharing), continuous batching | Excellent ecosystem integrations, Modelfiles, official Docker image, optional cloud models |

#### When Ollama is usually better: You want a fully open-source, auditable tool.

- You are a developer building scripts, agents, or apps (many frameworks and coding tools assume Ollama by default).
- You need Docker, Kubernetes, or long-running headless servers with minimal footprint.
- You prefer the lowest idle RAM and fastest cold-start times (especially on NVIDIA GPUs).
- You want the absolute simplest CLI workflow (ollama run model).

## Local-first inference

**RamaLama vs Ollama — key differences**

Both tools let you run large language models locally, but they take different approaches.

### Quick summary

| Aspect | **Ollama** | **RamaLama** |
|----|----|----|
| **Core idea** | Simple local model runner + library | Container-native way to run models |
| **Default execution** | Runs models directly on the host | Runs models **inside containers** (Podman/Docker) by default |
| **Who makes it** | Ollama Inc. | containers org (Podman / Red Hat ecosystem) |
| **Model sources** | Primarily its own library (+ limited HF) | Hugging Face, Ollama registry, ModelScope, any OCI registry, URLs, local files |
| **Runtimes** | Built around llama.cpp | llama.cpp (default) **and** vLLM |
| **Hardware handling** | You (or the installer) handle GPU libraries | Auto-detects GPU and pulls a matching optimized container image |
| **Security / isolation** | Host process | Rootless containers, read-only mounts, network isolation by default |
| **Commands feel like** | `ollama run`, `ollama pull`, etc. | Familiar container-style: `ramalama run`, `pull`, `serve`, `push` |
| **Best for** | Easiest “just chat with a local model” experience | People who already use containers and want the same workflow + stronger isolation |

### Main philosophical difference

- **Ollama** optimizes for simplicity and a polished consumer experience. Install it, pull a model, chat. Models usually run as normal processes on your machine.
- **RamaLama** optimizes for *making AI work like containers*. It treats models more like images: pull from multiple registries, run them in isolated containers, serve them, convert them to OCI images, push them to registries, etc. The slogan is essentially “make working with AI boring” by using the tools and patterns people already know from Podman/Docker.

### Practical implications

- **Isolation & security**: RamaLama wins by default (models stay inside rootless containers).
- **GPU setup**: RamaLama is often easier on Linux (especially Fedora/RHEL-style systems) because the container already contains the right CUDA/ROCm/etc. libraries.
- **Model choice**: RamaLama is more flexible about where models come from (especially Hugging Face and OCI registries).
- **Ecosystem fit**:
  - Ollama → very widely supported by agents (goose, OpenCode, Hermes, etc.).\
  - RamaLama → also usable (many tools treat it as an OpenAI-compatible endpoint or have explicit support), and it can even pull from the Ollama registry.

### When to choose which

- Choose **Ollama** if you want the simplest, most popular local model runner and don’t care about container isolation.
- Choose **RamaLama** if you already live in a container world (Podman/Docker), want better isolation by default, or need to pull models from Hugging Face / OCI registries more cleanly.

*Both expose OpenAI-compatible APIs, so most agents and tools can talk to either of them.*

### Other local inference options

- **llama.cpp** offers low-level control and GGUF support.
- **LM Studio** provides a desktop interface and API.
- **Jan** is an open-source desktop application.
- **Open WebUI** provides a self-hosted web frontend.
- **KoboldCpp** and **vLLM** support additional local and high-throughput serving setups.

Model management includes downloading and selecting models, quantization, Modelfiles, and custom system prompts. Hardware, memory requirements, and trade-offs vary by model and runner.

## Cloud and hybrid options

Hosted providers can complement local inference when a task needs a more capable model or additional capacity. Hybrid setups balance privacy, cost, latency, rate limits, and data-sovereignty requirements.

## Making models usable by agents

## Security and operational considerations

### Security Resources for local models

- 2026-10-21 [Kai: Your Local AI Is Not as Private as You Think](https://youtu.be/3mx84MRj8lY?si=ffjk7oazRh3vbYrr)
- 2026-10-19 [Cloud Codes: Your Local AI Model Isn’t as Safe as You Think](https://youtu.be/7S3kCU1tXcA?si=ABvWS3BS3Og4Va72)

### Local Model Vulnerabilities

Here is an overview of the main known classes of LLM exploits and attack surfaces (as of late 2026). This focuses on categories that appear repeatedly in research, real incidents, and frameworks such as the **OWASP Top 10 for LLM Applications (2026 edition)**. It deliberately stays at the conceptual level.

#### 1. Prompt Injection (still \#1)

- **Direct**: Attacker writes instructions that override the system prompt or safety rules.
- **Indirect / context injection**: Malicious instructions are planted in external content the model later processes (emails, documents, web pages, RAG results, tool outputs, memory, etc.). Real-world examples include zero-click data exfiltration (e.g., EchoLeak against Microsoft 365 Copilot) and tool/agent hijacking.
- Variants include jailbreaks (role-play, encoding, multi-turn persuasion) aimed at bypassing safety training.

#### 2. Sensitive Information Disclosure / Data Leakage

Models or applications can leak training data, system prompts, user conversation history, credentials, or proprietary information through responses, logs, embeddings, or tool calls. Techniques include membership inference, training-data extraction, and system-prompt extraction.

#### 3. Excessive Agency / Tool & Agent Abuse

When an LLM can call tools, browse, write code, send messages, or control systems, injected instructions can make it perform unauthorized actions (file access, API calls, privilege escalation, lateral movement). This risk rose sharply in rankings with the spread of agentic systems.

#### 4. Supply-Chain Attacks

- Malicious models or weights (including the pickle / “broken pickle” / Sleepy Pickle family we discussed earlier).
- Poisoned packages, MCP servers, plugins, or fine-tuning artifacts.
- Compromised model hubs or distribution channels.

#### 5. Data & Model Poisoning / Backdoors

Attackers influence training, fine-tuning, or retrieval data so the model behaves normally most of the time but activates harmful behavior under a trigger (sleeper agents, backdoored code generation, biased outputs, etc.).

#### 6. Unbounded Consumption / Resource Abuse

Attacks that force excessive token generation, repeated inference, or expensive tool use, driving up cost or causing denial-of-service. Also includes model-extraction / distillation attacks that steal capability by querying the target model.

#### 7. Misinformation / Hallucination Abuse

Models can be steered (or simply fail) into producing confident but false or harmful content that users or downstream systems act upon.

#### 8. Hidden Context / System-Prompt Exposure

Secrets, authorization logic, or internal instructions placed in the system prompt or retrieved context can be extracted or abused.

#### 9. Vector / Embedding / RAG Weaknesses

Poisoning of vector stores, cross-tenant leakage in embeddings, or adversarial retrieval that surfaces malicious documents.

#### 10. Improper Output Handling

Model output is treated as trusted and fed into browsers, shells, databases, or other interpreters without sanitization, enabling classic injection or XSS-style follow-on attacks.

#### Additional established categories outside the strict Top 10

- **Adversarial examples / input perturbations** — small, often imperceptible changes that flip model behavior.
- **Model inversion & privacy attacks** — reconstructing training examples or attributes.
- **Multimodal attacks** — injecting payloads via images, audio, or other modalities that the model processes.
- **Agent-specific chains** — memory poisoning, persistent tool hijacking, self-propagating behavior across sessions or systems.
- **Infrastructure / serving vulnerabilities** — flaws in inference servers, gateways (e.g., LiteLLM-style issues), or serialization formats beyond pickle.

#### Practical note

Most real-world incidents combine several of the above (e.g., indirect prompt injection → excessive agency → data exfiltration). Open-weight models have repeatedly shown higher susceptibility to certain cyber-relevant payload chains than the strongest closed models, though both remain targets.

The landscape evolves quickly; the dominant practical risks in production systems right now center on **prompt/context injection**, **agent tool abuse**, and **supply-chain model/package poisoning**. Defenses typically involve layered controls: input/output filtering, least-privilege tooling, provenance checks, retrieval isolation, and continuous red-teaming rather than relying on any single technique.

## **Security Summary: What Can the Model Reach?**

Running models locally protects your prompts from third-party providers, but it does **not** automatically make the setup safe. The decisive security question is not “Is the model local?” — it is **“What can the model (or the software around it) reach?”**

### The practical risk ladder

| Reach level | What the model can touch | Risk profile |
|----|----|----|
| **Chat only** | Conversation text | Low — limited to generating text |
| **Files / project context** | Local documents, code, `.env` files | Medium — can exfiltrate or rewrite secrets if it has write access |
| **Tools / shell / network** | Terminal commands, HTTP requests, file system, APIs | High — can execute code, send data out, or fetch remote payloads |
| **Agent + persistent access** | Multiple tools + memory + long-running privileges | Critical — full blast radius of the host |

A backdoored or triggerable model (or a poisoned chat template) is only as dangerous as the permissions and tools you give it. In the ProjectDiscovery demonstration highlighted in the video, a fine-tuned model behaved normally on clean tests until a short trigger phrase caused it to instruct an agent to send the project’s `.env` file to an external server. The model itself did not contain the final payload; it simply knew to fetch and run one.

### Key attack surfaces that expand “reach”

- **Model file + chat template** — Weights are data, but GGUF templates (Jinja) and metadata can embed hidden instructions or, in rare cases, enable code execution if the runner is unsandboxed.
- **Runner / inference server** — Bugs or misconfigurations in Ollama, LM Studio, llama.cpp, etc., can leak memory, environment variables, or allow unauthorized control.
- **Network binding** — Binding to `0.0.0.0` (or leaving the port unauthenticated) turns a local server into a public one. Exposed, unauthenticated instances are routinely scanned and abused.
- **Tool / agent permissions** — Shell access, file write, or unrestricted network is the difference between a chat bot and a remote-code-execution surface.

### Practical containment principles

1.  **Minimize reach by default** — Prefer pure chat. Grant tools only when necessary and under least privilege.
2.  **Keep the server on loopback** (`127.0.0.1`). For remote access use an authenticated tunnel (e.g., Tailscale) or a reverse proxy with authentication — never expose the raw inference port.
3.  **Keep the runner updated** and enable any available authentication features.
4.  **Treat model files as untrusted input** — Prefer reputable sources, verify provenance where possible, and consider scanning templates.
5.  **Separate secrets from the model’s working environment** — Do not give the process access to credentials it does not need.

**Bottom line:** Local inference removes the cloud provider from the trust boundary, but it transfers the entire security responsibility to you. The model is only as private and safe as the smallest set of resources you allow it to reach.

## NVIDIA \[OpenShell\]\[(https://www.nvidia.com/en-us/ai/openshell/) is an open, secure runtime designed for AI agents. It provides a governed environment that controls agent execution, access, and inference routing, independent of the underlying model or application.

### Core Idea

Security is enforced in the environment rather than inside the model or agent code. Nothing is allowed by default. Permissions are granted through explicit policies, and enforcement occurs outside the agent’s process so it cannot be bypassed or “prompted away.” All allow/deny decisions are fully auditable. It works with any model, agent harness, or framework across cloud, hybrid, on-premises, edge, and air-gapped environments under a single policy layer.

### Key Benefits

- **Agent governance outside the model**: Separates agent behavior from environment interactions; every action is monitored and controlled.
- **Zero-trust access**: Deny-by-default model with intent-based permissions enforced externally.
- **Safe operation of agent fleets**: Unified policy enforcement for open or closed models and various harnesses at scale.

### Main Components

- **Agent Sandboxes**: Isolated environments with no direct network access, privilege restrictions, file limits, and kernel-level system-call filtering. Requests are brokered through a secured channel.
- **Policy Prover**: Formal verification engine that checks policies stay within allowed boundaries and evaluates network rules for risk.
- **Gateway**: Control plane that authenticates users, manages sandbox lifecycles, delivers policies/credentials, and brokers access.
- **Supervisor**: Runtime overseer outside the sandbox that evaluates network requests against policy (at binary, destination, method, and path levels), supplies credentials only when allowed, applies live policy updates, and logs every decision.

### Reference Architecture

A full-stack setup powered by the OpenShell runtime, NVIDIA Vera CPUs, Vera Rubin systems, and NVIDIA Sentry on BlueField-4 for industrial-scale safe agents in AI factories, enterprise settings, and at the edge.

### Who It’s For & What It Supports

Primarily developers building autonomous agents, platform teams, and security/IT teams that need to enable agent use without granting broad host, network, credential, or model access. It is model- and harness-agnostic (supports Claude Code, Codex, GitHub Copilot CLI, LangChain Deep Agents, custom agents, etc.) and is not itself an agent framework—it sits underneath them. It goes beyond simple containers/pods by adding agent-specific controls for lifelong governance, credential handling, egress policy, inference routing, and multi-agent workflows.

OpenShell is open source (available on GitHub), runs on Docker, Podman, Kubernetes (via Helm), and experimental runtimes, works with NVIDIA DGX systems, supports cloud model endpoints via a policy-aware router, and does not require Vera CPUs (though they provide performance and efficiency gains).

### Technical Resources

- [NVIDIA YouTube Summary](https://youtu.be/GYYP-eW58ug?si=cEg8YqMXA8lZ26c-)
- [NVIDIA Developer: How NVIDIA OpenShell Secures Autonomous Agents \| Nemotron Labs](https://www.youtube.com/live/izFGht5VDL8?si=z6VrOgSXwqWB-jZz)

Back to top
