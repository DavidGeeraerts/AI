# Prompt Engineering Notes

Published

2026-10-27

# Key Takeaways

- [tags](https://github.com/DavidGeeraerts/AI/issues/8#issuecomment-6445792490) (especially XML-style) are a widely recommended best practice in modern prompt engineering, and prompts are commonly stored as plain Markdown (.md) files (or similar text files) for version control, readability, and maintainability.
  - XML-style tags for unambiguous, explicit boundaries and nesting. Closing tags make it clear where a section ends

## Example of a Best-Practice Prompt using XML-style tags

``` xml

<role>
expert at...
</role>

<context>
    - The context provides the necessary background information for the prompt.
</context>

<instructions>
1. Do this
2. Do that
3. Do the other thing
</instructions>

<rules>
- Always follow the instructions.
- Never generate text that violates the rules.
</rules>

<examples>
    <example>
        - This is an example of a good prompt.
    </example>

    <example>
        - Nesting tags is possible, e.g., for a list of examples:
    </example>
</examples>

<output_format>
- markdown table

</output_format>
```

# Resources - YouTube

| Title | Rating | Notes |
|----|----|----|
| [Mastering Claude Code in 30 minutes](https://www.youtube.com/live/6eBSHbLKuN0?si=wN2J_RYKiefkdcMH) | Low | High noise low signal. Infomercial. |
| [The prompting playbook](https://youtu.be/G2B0YWuJUgI?si=SCAOnfZsPJBtr3QS) | Medium | Good for understanding the fundamentals of prompting. |
| [Google’s 9 Hour AI Prompt Engineering Course In 20 Minutes](https://youtu.be/p09yRj47kNM?si=YCaSduB3VAEhvqSK) | medium | A good overview of the basics of prompt engineering. |
| [DeepLearningAI](https://youtu.be/8ib4Qnh2HFE?si=QULwzDGBp30T7-Jr) | High | A comprehensive course on prompt engineering. |

## [Anthropic: Mastering Claude Code in 30 minutes](https://www.youtube.com/live/6eBSHbLKuN0?si=wN2J_RYKiefkdcMH) [Boris Cherny](https://borischerny.com/)

- create an AGENTS.md file to give the agent context about the project, including instructions, constraints, and goals.

## [Claude: The prompting playbook](https://youtu.be/G2B0YWuJUgI?si=SCAOnfZsPJBtr3QS) \[Margot van Laar\]

- Use XML tags to structure prompts for clarity and consistency.
- Be clear and direct in your instructions, avoiding ambiguity.
- Curate the information you provide to the model, focusing on relevant details.
- Start with an evaluation suite before changing any prompt. Cover three types of cases: control cases (should always pass), edge cases (previous failures you want to prevent), and capability-boundary cases (when the model should escalate, refuse, or hand off to a human).
- Apply prompt hygiene first: remove redundant or outdated content (e.g., website copy, cookie notices, “you are a human” statements) and separate role, guidelines, policy, tone, and data with clear structure (XML tags work especially well).
- If a human cannot distinguish guidelines from policy from data while reading the prompt, the model almost certainly cannot either.
- Create an explicit output contract (e.g., wrap the response in specific XML tags) and reinforce it at the harness level with stop sequences so the model knows exactly when to stop.
- Treat old defensive patches as technical debt. Instructions written for a previous model’s weaknesses often cause new failures (including the opposite of hallucination: withholding information the model actually has). Version-control the *reason* for every instruction so you can remove it when the model improves.
- Instructions do not add capability. Telling the model “calculate correctly” or “this is critical” does not improve arithmetic. Give it a reliable tool (with proper API declaration, schema, and implementation) instead, and let the model reason about when to use it.
- When the model must make a trade-off (e.g., escalate a billing error vs. keep handling the conversation), state *both* sides of the cost/benefit clearly. Stronger models will otherwise overfit to whichever side you emphasize.
- When building a new agentic system, optimize the combination of model × prompt × harness (tools, structured outputs, stop sequences, adaptive thinking, etc.), not the prompt in isolation.
- For multi-step or constrained generation tasks, prefer a generate → evaluate → repair loop with several simple, focused prompts over one giant monolithic prompt.

## [Tina Huang: Google’s 9 Hour AI Prompt Engineering Course In 20 Minutes](https://youtu.be/p09yRj47kNM?si=YCaSduB3VAEhvqSK) [Tina Huang](https://www.linkedin.com/in/tinahuang/)

- Use the five-step TCREI framework for every prompt: **Task** (what to do, plus persona + desired format), **Context** (background, audience, constraints), **References** (examples / few-shot), **Evaluate** (does the output meet the need?), **Iterate** (refine and retry). Mnemonic: “Tiny crabs ride enormous iguanas” (or Google’s “Thoughtfully Create Really Excellent Inputs”).
- Always be iterating (“ABI”). Four practical iteration methods: (1) revisit the full TCREI framework and add missing pieces, (2) break long prompts into shorter, clearer sentences, (3) rephrase or try an analogous task, (4) add constraints to narrow the solution space.
- Specify persona and output format early in the Task step—they dramatically improve relevance and usability.
- Provide rich context and concrete references/examples; more relevant detail almost always beats a sparse prompt.
- Multimodal prompting follows the same principles: clearly state what the inputs are (text, image, audio, etc.) and what form the output should take.
- Watch for hallucinations and inherited biases. Keep a human-in-the-loop: always verify important outputs before using or sharing them. Avoid feeding sensitive data into public models.
- For everyday work: build and maintain a personal prompt library of reusable, high-performing prompts (emails, summaries, tables, brainstorming, etc.).
- For data analysis & presentations: prompt for insights, formulas, patterns, and slide structures, but treat generated numbers and claims with extra scrutiny.
- Advanced techniques:
  - **Prompt chaining** — break a complex goal into a sequence of linked prompts, feeding each output into the next (e.g., manuscript → summaries → taglines → full marketing plan).
  - **Chain-of-Thought (CoT)** — explicitly ask the model to show its step-by-step reasoning.
  - **Tree-of-Thought (ToT)** — ask the model to explore multiple reasoning paths / options in parallel and evaluate them.
  - **Meta-prompting** — ask the AI to help write or improve the prompt itself when you are stuck.
- Create lightweight AI agents by defining a clear persona, context, conversation rules, and a stop phrase. Two useful patterns: simulation agents (role-play / practice interviews) and expert-feedback agents (critique pitches, drafts, etc.).
- Responsible use checklist: verify accuracy, check for bias, protect sensitive information, and treat AI as a collaborator rather than an infallible authority.

## [DeepLearningAI: Prompt Engineering for Developers](https://youtu.be/8ib4Qnh2HFE?si=QULwzDGBp30T7-Jr) [DeepLearningAI](https://www.deeplearning.ai/) [Andrew Ng](https://x.com/AndrewYNg)

- Treat the AI like a smart, highly motivated fresh college graduate who knows nothing about you yet. Give it the context a new hire would need (documents, screenshots, project trackers, voice memos, past work) instead of short Google-style queries.
- Provide rich, relevant context. Modern models can handle very large context windows (~750k words). Start a new chat for unrelated topics so stale context does not degrade performance.
- Understand the three knowledge sources and when to use each:
  - Pre-trained knowledge (good for general/common facts; limited by training cutoff and data frequency).
  - Web search (real-time info; note that the main model often only sees summaries produced by a helper model, not the full pages).
  - Deep research / agentic modes (many parallel searches, iterative evaluation, synthesized cited reports—for complex, multi-source questions).
- Defeat sycophancy (the model’s tendency to please you). Ask neutral questions that do not hint at your preferred answer. Use objective rubrics with clear scoring criteria *before* asking for a score or judgment.
- Use AI as a thought partner, not a one-shot answer machine. For brainstorming: supply context → request multiple options → give feedback → iterate. Feedback on options often matters more than more upfront context.
- Prefer progressive outlining for writing: outline → critique/refine the outline → expand to bullet points → iterate → only then draft full text. This avoids generic “AI slop” and makes structural changes easy.
- For complex reasoning tasks, explicitly cue longer thinking (“think hard,” “ultra think,” or enable reasoning/thinking mode). Classic “think step by step” is largely obsolete on modern models.
- When editing or evaluating work, give the model a clear rubric first; then ask it to apply the rubric. This produces more objective, useful feedback.
- Leverage multimodal and tool-using capabilities: upload images/files, generate images, analyze data/spreadsheets, write and run code, and build simple websites/apps with natural-language instructions (no coding background required).
- Desktop/agentic AI apps can explore local files (with permission). Best practice: state the overall task, let the model propose a plan, then approve execution.
- Power-user habits: iterate aggressively, request multiple options, keep conversations focused, verify important claims (especially from search/deep research), and continually give the model harder, more context-rich tasks.

# Articles & Blogs

- [The Prompt Is Not the Architecture — But It Still Governs How the System Reasons](https://interestingengineering.substack.com/p/the-prompt-is-not-the-architecture)

## [The Prompt Is Not the Architecture — But It Still Governs How the System Reasons](https://interestingengineering.substack.com/p/the-prompt-is-not-the-architecture) (Interesting Engineering, 2026-05-23)

- The prompt is not the architecture, but it still governs *how* the system reasons. Architecture (harness, tools, gates, loops, meta-observation) determines what the system can see and do; the prompt/spec governs reasoning style and action selection.
- Prompting has fragmented into four distinct disciplines (per Nate B Jones). Most people only practice the first:
  1.  Prompt craft (structure, XML, role, output format)
  2.  Context engineering (domain facts, background, memory tiers)
  3.  Intent engineering (state the desired outcome / what “good” looks like, not just the task)
  4.  Specification engineering (acceptance criteria, constraints, ordering, trade-off framings)
- A strong single structured system prompt that embeds all four disciplines can outperform multi-agent swarms on single-turn, well-scoped document tasks (H2 scored α = 1.000; five-agent swarm scored 0.625).
- Brownfield (existing, messy production systems) vs greenfield (clean-slate tasks) matters. Complex constraint-satisfaction or continuous operational systems usually need correction loops or adaptive architectures, not just a bigger prompt.
- Generate → Evaluate → Repair (GER) loops are often more reliable and cheaper than relying solely on the model’s adaptive/extended thinking for hard constraint tasks.
- Instructions do not add capability. Arithmetic, deterministic calculations, and similar tasks should be offloaded to tools/functions rather than instructed in natural language.
- Legacy defensive patches written for older, weaker models can actively harm performance on newer models (e.g., causing the model to withhold information it actually has). Audit and remove outdated guardrails on model upgrades.
- Prefer trade-off framings over rigid binary rules for judgment calls (state both sides of the cost/benefit). Keep hard binary checks only at verification gates.
- Use API-level stop sequences / output contracts (not just prompt text) to enforce structured endings and reduce padding or runaway generation.
- Ordering of instructions matters: place steps in the same sequence a competent human would reason through the problem.
- The “35-minute wall”: once agents run autonomously for roughly that long without check-ins, chat-style prompting assumptions break. Everything critical must be pre-encoded in the specification before launch.
- Intent engineering (closing the loop between predicted outcomes and actual results, then calibrating future behavior) is often the missing operational discipline, not a pure engineering problem.
- Meta-harness / meta-observation layers that read execution traces across runs can surface failure modes that pure pre-specification cannot anticipate.
- Architecture choices should match the problem type: high-density single prompt for greenfield document tasks; correction loops + tools + meta-observation for brownfield continuous operations.

Back to top
