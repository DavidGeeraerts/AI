# Agents and Harnesses

Published

2026-10-26

## Introduction

- **An agent** is the autonomous actor — the thing that reasons, plans, uses tools, remembers, and pursues goals over multiple steps.
- **A harness** is the runtime / scaffolding that turns a raw language model into a working agent.
  - ***The harness** is the engine and chassis. **The agent** is the driver + the behavior that emerges once the model is running inside that chassis.*

## Agent Harness Diagram

![](./assets/images/agent-harness-diagram.png "Agent Harness Diagram")

Agent Harness Diagram

### More precise breakdown

| Concept | What it is | Examples |
|:---|:---|:---|
| **Model** | The raw intelligence (weights) | Qwen, DeepSeek, Claude, Llama, Ajax (abliterated), etc. |
| **Harness** | The runtime that gives the model a loop, tools, memory, sandboxes, skills, messaging, persistence, etc. | Hermes, OpenClaw, goose, OpenHands, OpenCode, Agent Zero |
| **Agent** | The resulting autonomous system that actually does work | Copilot, OpenCode, Agent Zero |

**The harness determines:** - How long the agent can run - Whether it has persistent memory and can create its own skills - What tools (MCP servers) it can reach - Whether it lives in messaging apps, a terminal, a web UI, or a full Linux desktop - How isolated / sandboxed its actions are - How easy it is to observe, interrupt, or improve it

A good harness {Hermes, OPenCode, Agent Zero, OpenClaw, goose, etc.} dramatically changes what kind of agent you can actually have. A weak or missing harness leaves you with just a chatbot that forgets everything between messages

A harness is the runtime that gives a model a loop, tools, memory, and an environment in which to act. It determines how long an agent can run, what it can access, and how its work can be observed and controlled.

## Modern harnesses

| Name | Type | Primary Focus | Notes / Key Traits | Stars (approx, Oct 2026) |
|----|----|----|----|----|
| **goose** | Both | General-purpose local agent | Own harness + agent; MCP-native; desktop + CLI; AAIF | ~55k |
| **Hermes Agent** | Both | Always-on personal agent | Own runtime; persistent memory + skill creation; messaging gateways | ~250k+ |
| **Agent Zero** | Both | Full computer-use agent | Own Docker Linux desktop + agent; high transparency | ~19.5k |
| **OpenClaw** | Both | Personal assistant (chat apps) | Own gateway/runtime; messaging-first | Very high (~390k+) |
| **OpenHands** | Both | Coding / software engineering | Own multi-agent runtime; web UI + desktop | ~89k |
| **OpenCode** | Both | Terminal coding agent | Own agent loop; model-agnostic | High |
| **Codex** (OpenAI) | Both | Coding agent suite | Cloud + CLI + IDE; proprietary harness | N/A (product) |
| **Claude Code** | Both | Coding agent | Anthropic harness + agent; subscription-based | N/A (product) |
| **Cline** | Both | IDE coding agent | VS Code / JetBrains extension with own loop | — |
| **Aider** | Both | Terminal coding agent | Own agent loop; git-aware | — |
| **Eigent** | Both | Multi-agent desktop | Parallel agents on one task | ~15.5k |

Compare harnesses by their philosophy, strengths, limitations, deployment style, and fit for the task.

## Supporting infrastructure

- **MCP (Model Context Protocol)** connects agents to tools and services.
- Sandboxed execution environments such as Docker, Daytona, Modal, E2B, and browser infrastructure help isolate agent actions.
- Persistent memory and agent-written skills support longer-running work.
- **SpyNEL** orchestrates existing coding harnesses.
- **DOX / AGENTS.md** hierarchies give agents maintainable project context.

## [agents.md](https://agents.md/)

#### Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?

[Evaluating AGENTS.md: Are Repository-Level Context Files Helpful…](https://arxiv.org/abs/2602.11988)

| Condition | Effect on success rate | Effect on cost / steps |
|----|----|----|
| LLM-generated context files | Slightly worse on average (~ –0.5% on SWE-bench Lite, ~ –2% to –3% on the new bench; negative in most configs) | +20–23% inference cost; +2.5–4 extra steps per task |
| Human-written context files | Modest average gain (~ +4% on the developer-file benchmark) | Still + up to ~19% cost / more steps |
| Overall | “Does not generally improve task success rates” | Clear, consistent cost increase |

Practical takeaway that matches the evidence: Keep any AGENTS.md / CLAUDE.md extremely short. Prefer only non-discoverable facts (exact test/build commands that differ from the obvious, hard “never touch X”, unusual tooling). Long files that restate what is already in the code, README, or package.json are the ones most likely to waste tokens for little or no gain.

Back to top
