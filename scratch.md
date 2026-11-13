This is the real world. Most tutorials trap you in a walled garden—a simulation built by cloud vendors to hide the metal. We are going to strip away the abstractions and build an agentic stack that you actually control. No corporate speak. No magic APIs. Just you, the hardware, and the models.

Because this "course" is entirely contained within **Chapter 05: Building a Working Stack** of *The Book of AI*, we will build it using Quarto's `{{< include >}}` directive. This keeps your master chapter clean while breaking the matrix of code down into manageable, modular files.

Here is the directory structure, the master file, and the deeply detailed, hierarchical blueprints for Blocks 0 through 6.

---

### The Quarto Directory Structure

Inside your `DavidGeeraerts/AI` repository, you will create a `blocks/` directory next to `chapters/` to hold the included modules for Chapter 5.

```text
AI/
├── _quarto.yml
├── chapters/
│   ├── 01-introduction.qmd
│   └── 05-building-a-working-stack.qmd    <-- The Master File
├── blocks/
│   ├── ch05_block_0_preflight.qmd
│   ├── ch05_block_1_fundamentals.qmd
│   ├── ch05_block_2_models.qmd
│   ├── ch05_block_3_running_models.qmd
│   ├── ch05_block_4_agents_harnesses.qmd
│   ├── ch05_block_5_orchestrating_swarm.qmd
│   └── ch05_block_6_utilities.qmd
└── appendices/
    └── glossary.qmd

```

---

### The Master File: `chapters/05-building-a-working-stack.qmd`

This file is the construct. It loads the environment.

```markdown
# Chapter 05: Building a Working Stack

You have a machine. It is time to wake it up. 

In this chapter, we build an agentic engineering stack from the bare metal up. We do not rely on hosted platforms that hide the mechanics. You have to understand the rules of the system so you can bend them. We will start with the operating system, load the editor, download the weights, and build the harnesses that turn static code into autonomous operators.

{{< include ../blocks/ch05_block_0_preflight.qmd >}}

{{< include ../blocks/ch05_block_1_fundamentals.qmd >}}

{{< include ../blocks/ch05_block_2_models.qmd >}}

{{< include ../blocks/ch05_block_3_running_models.qmd >}}

{{< include ../blocks/ch05_block_4_agents_harnesses.qmd >}}

{{< include ../blocks/ch05_block_5_orchestrating_swarm.qmd >}}

{{< include ../blocks/ch05_block_6_utilities.qmd >}}

```

---

### Block 0: Pre-Flight (`blocks/ch05_block_0_preflight.qmd`)

```markdown
## Block 0: Pre-Flight

Before we load the models, we need a stable construct. The operating system is your base reality. 

```{mermaid}
graph TD
    Hardware["Bare Metal"] --> Kernel["Linux Kernel"]
    Kernel --> UserSpace["User Space (mise, uv, Docker)"]
    UserSpace --> Editor["The Editor (VS Code)"]
    Editor --> Workspace["Project: Agentic Stack"]

```

### Lab 0.1: GNU/Linux distribution choices

The kernel is the same, but the package managers dictate how you interact with the system. Choose your branch.

#### RHEL branch

* **Lab 0.1.1: RHEL** - The enterprise standard. Slow, stable, unyielding. We configure `dnf` and SELinux.
* **Lab 0.1.2: Fedora** - The bleeding edge of the Red Hat lineage. We pull the latest kernel for better GPU driver support.
* **Lab 0.1.3: CentOS Stream** - The mid-ground. Testing the waters before RHEL.

#### Debian branch

* **Lab 0.1.4: Debian** - The universal operating system. Rock solid. `apt` is your interface.
* **Lab 0.1.5: Ubuntu** - The mainstream path. Heavy, but everything works out of the box.
* **Lab 0.1.6: Mint** - The comfortable desktop. We strip away the GUI bloat to find the terminal.

#### Arch branch

* **Lab 0.1.7: Arch** - You build it yourself. `pacman` and the AUR. Maximum control, maximum responsibility.
* **Lab 0.1.8: Omarchy** - DHH's opinionated tiling setup. Hyprland and Quickshell. Fast, keyboard-driven, built for operators.
* **Lab 0.1.9: Manjaro** - Arch with training wheels.

#### Gentoo branch

* **Lab 0.1.10: Gentoo** - Compiling everything from source. You control the exact flags for your CPU architecture. Not for the faint of heart.

#### SUSE branch

* **Lab 0.1.11: openSUSE** - `zypper` and YaST. The chameleon.

#### Alpine branch

* **Lab 0.1.12: Alpine** - `apk` and `musl` libc. Ultra-lightweight. We use this primarily for our containerized sandboxes later.

### Lab 0.2: VS Code installation and configuration

Your editor is your jack-in point. We configure it to be fast, dark, and deeply integrated with the terminal.

* **Lab 0.2.1: Base installation** - Pulling the binary. Disabling telemetry. We don't need Microsoft tracking our keystrokes.
* **Lab 0.2.2: Extension installation** - Loading the modules we need: `Python`, `Quarto`, `Ruff` (for fast linting), and `Docker`.
* **Lab 0.2.3: Theme and settings** - Setting a high-contrast dark theme. Binding `Ctrl+~` to the integrated terminal.

### Lab 0.3: Runtime Isolation

System packages will break your stack. We isolate our runtimes in user-space.

* **Lab 0.3.1: Installing `mise**` - The tool manager. We bypass `/usr/bin/` entirely.
* **Lab 0.3.2: Pinning `uv` and Python** - Setting Python 3.12 as our global standard. Fast virtual environments.

```

Watching someone navigate a heavily customized Linux terminal and tiling window manager makes the concepts click faster than reading commands.

<Carousel>
  <Video component_placeholder_id="youtube_video_0" count="3" query="Linux terminal workflow tiling window manager VS Code" source="youtube"/>
</Carousel>

---

### Block 1: Introduction & Fundamentals (`blocks/ch05_block_1_fundamentals.qmd`)

```markdown
## Block 1: Introduction & Fundamentals {Building Blocks}

Code used to be a strict set of tracks. A machine following a path. We are introducing stochastic brains into deterministic systems. 

```{mermaid}
flowchart LR
    Task["Task"] --> Router{"Is it predictable?"}
    Router -->|Yes| Script["Standard Python Code<br/>(Fast, Cheap, 100% Reliable)"]
    Router -->|No| Agent["Agentic LLM Loop<br/>(Slow, Expensive, Adaptive)"]

```

### Lab 1.1: The nature of an agent

* **Lab 1.1.1: Deterministic code** - We write a Python script to parse a log file. It works perfectly until the log format changes by one character. It breaks.
* **Lab 1.1.2: Stochastic execution** - We pass the broken log to a raw LLM prompt. It understands the intent and extracts the data, but the output format is unpredictable.

### Lab 1.2: The contract layer

To bridge the gap, we build contracts. We force the stochastic brain to communicate in strict, typed structures.

* **Lab 1.2.1: Defining the input schema** - What exactly does the agent need to know?
* **Lab 1.2.2: Pydantic outputs** - We write the JSON schema that the LLM must obey to pass data back to our system.

```

---

### Block 2: Models (`blocks/ch05_block_2_models.qmd`)

```markdown
## Block 2: Models

The models are the weights. Matrices of numbers that compress human knowledge. We need to understand how they are built and how to shrink them so they run on our local hardware.

### Lab 2.1: Model architectures
*   **Lab 2.1.1: The Transformer** - Stripping down attention mechanisms. How a token looks at every other token to find context.
*   **Lab 2.1.2: MoE (Mixture of Experts)** - How models route queries to specialized sub-networks to save compute.

### Lab 2.2: Model weights and quantization
You cannot run a 70-billion parameter model in FP16 on a laptop. We have to compress it.
*   **Lab 2.2.1: File formats** - The difference between `Safetensors` (Python/cloud) and `GGUF` (local/C++).
*   **Lab 2.2.2: Quantization levels** - We benchmark a model at FP16, Q8 (8-bit), and Q4_K_M (4-bit). We measure the exact point where brain-damage (loss of logic) occurs.

### Lab 2.3: Context windows
*   **Lab 2.3.1: Tokenization** - We write a script to see how the machine actually reads words. "Agentic" becomes `['Agent', 'ic']`.
*   **Lab 2.3.2: Context rot** - The "Lost in the Middle" phenomenon. We load a 16k context window and watch the model forget instructions placed in the center of the prompt.

```

---

### Block 3: Running models (`blocks/ch05_block_3_running_models.qmd`)

```markdown
## Block 3: Running models {locally, hosted, cloud}

A model on a hard drive is dead data. We need an inference engine to pump electricity through the weights.

```{mermaid}
graph TD
    Client["Your Python Code"] --> Interface["HTTP / OpenAI API Format"]
    Interface --> Local["Lab 3.1: Local (llama.cpp)"]
    Interface --> Hosted["Lab 3.2: Self-Hosted (vLLM)"]
    Interface --> Cloud["Lab 3.3: Cloud API (Anthropic/OpenAI)"]

```

### Lab 3.1: Local inference (Bare Metal)

* **Lab 3.1.1: Ollama daemon** - The easiest way to pull and run weights. We start the service and watch it allocate VRAM.
* **Lab 3.1.2: llama.cpp server** - Dropping down a level. We compile `llama-server` from source to control flash-attention and thread counts manually.

### Lab 3.2: Hosted containers (The Grid)

* **Lab 3.2.1: vLLM** - When one request at a time isn't enough. We spin up a Docker container with vLLM to handle continuous batching and high throughput.
* **Lab 3.2.2: SGLang** - Exploring alternative backends for faster prefix-caching.

### Lab 3.3: Cloud APIs (The outside world)

* **Lab 3.3.1: LiteLLM routing** - We build a proxy router. If our local model fails or times out, we seamlessly route the request to a frontier cloud model without changing our application code.

```

---

### Block 4: Agents and Harnesses (`blocks/ch05_block_4_agents_harnesses.qmd`)

```markdown
## Block 4: Agents and Harnesses

A model is just a text predictor. An agent is a loop that gives the model eyes, hands, and memory. The harness is the code that keeps the loop from going rogue.

### Lab 4.1: Cognitive loops
*   **Lab 4.1.1: The ReAct loop** - We write a `while` loop. The model outputs a *Thought*, picks an *Action*, and we feed it back the *Observation*. 
*   **Lab 4.1.2: Loop breakers** - Agents hallucinate. We write the hard-coded circuit breakers that kill the loop if the agent repeats the same action three times.

### Lab 4.2: Tools and MCP
*   **Lab 4.2.1: Writing an MCP server** - We adopt the Model Context Protocol. We build a standalone server that grants the agent read-access to our file system.
*   **Lab 4.2.2: Tool binding** - We pass the MCP schema to the local LLM so it knows exactly how to format its JSON to pull the trigger on a tool.

### Lab 4.3: Memory systems
*   **Lab 4.3.1: Working memory** - Managing the sliding window. When the token count gets too high, we force the LLM to summarize its own past thoughts.
*   **Lab 4.3.2: Semantic memory** - We plug in a local vector database (Qdrant) so the agent can recall documents from last week.

```

---

### Block 5: Orchestrating the swarm (`blocks/ch05_block_5_orchestrating_swarm.qmd`)

```markdown
## Block 5: Orchestrating the swarm {Multi-Agent Systems}

One agent gets confused if you give it too many tools. The solution is division of labor. A swarm.

```{mermaid}
graph TD
    User["User Task"] --> Supervisor["Supervisor Agent"]
    Supervisor -->|Research Phase| Researcher["Search Agent (Read Only)"]
    Researcher -->|Context| Supervisor
    Supervisor -->|Execution Phase| Coder["Terminal Agent (Write Access)"]
    Coder -->|Result| Supervisor
    Supervisor --> User

```

### Lab 5.1: Topologies

* **Lab 5.1.1: Planner and worker** - We build a two-node graph. The Planner writes a JSON blueprint; the Worker executes it step-by-step.
* **Lab 5.1.2: State handoffs** - How to pass the working memory payload from one agent to another without losing context.

### Lab 5.2: Agent-to-Agent (A2A) protocols

* **Lab 5.2.1: Synthesizing context** - Agents shouldn't send raw HTML to each other. We write the compression prompts that allow agents to talk in dense summaries.

### Lab 5.3: Human-in-the-Loop (HITL)

* **Lab 5.3.1: The kill switch** - Before the Worker agent executes a terminal command, the harness pauses execution. It prints the command to your screen and waits for `[y/N]`. Never give a machine a loaded gun without a safety.

```

---

### Block 6: Misc tools and utilities (`blocks/ch05_block_6_utilities.qmd`)

```markdown
## Block 6: Misc tools and utilities

You are flying blind without telemetry. These are the tools operators use to see what the machine is actually thinking, and the armor used to contain it.

### Lab 6.1: Tracing and observability
*   **Lab 6.1.1: OpenTelemetry spans** - We wrap our agent loops in OTel spans. We watch the exact millisecond latency of a tool call vs. a generation phase.
*   **Lab 6.1.2: Visualizing the trace** - We spin up Arize Phoenix to see the waterfall graph of the agent's thought process.

### Lab 6.2: Sandboxing
*   **Lab 6.2.1: Rootless containers** - If the agent writes Python code, where does it run? We build a disposable, network-isolated Alpine container that executes the code and dies immediately. Total containment.

### Lab 6.3: Evals
*   **Lab 6.3.1: LLM-as-a-judge** - How do you know if a code change broke your agent? We write a script where a separate, larger model grades the swarm's output against a golden dataset.

```