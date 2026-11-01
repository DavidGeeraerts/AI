# Execution Methodology: Cost-Aware Routing & Escalation

## Core Directive
Maximize efficiency and minimize token costs by defaulting to the least expensive model capable of successfully completing the current task. Reserve frontier and reasoning models strictly for high-complexity cognitive work, architectural decisions, and deep debugging.

## Model Tiers & Delegation Rules

### Tier 1: Fast & Cost-Effective (e.g., Claude 3 Haiku, GPT-4o-mini, Gemini Flash)
**Default for all mechanical, repetitive, or strictly constrained tasks.**
*   **Data Parsing:** Extracting JSON, Regex application, parsing logs, or transforming data formats.
*   **Documentation:** Generating docstrings, inline comments, or formatting Markdown.
*   **Boilerplate:** Writing simple CRUD endpoints, CSS styling, or standard unit tests for isolated functions.
*   **File Operations:** Scanning directories or reading files to map imports.

### Tier 2: Balanced Workhorse (e.g., Claude 3.5 Sonnet, GPT-4o)
**Default for standard implementation and feature development.**
*   **Implementation:** Writing standard application logic, connecting APIs, and building UI components.
*   **Spec Generation:** Drafting `SPEC.md` or breaking down tasks into `TODO.md`.
*   **Standard Debugging:** Fixing clear runtime errors or linter warnings where the stack trace provides the answer.
*   **Code Review:** Performing standard PR reviews or security linting against known patterns.

### Tier 3: Frontier & Deep Reasoning (e.g., OpenAI o1, Claude 3 Opus)
**Strictly reserved for tasks requiring multi-step logic, vast context synthesis, or novel problem-solving.**
*   **Architecture:** Designing core system design, database schemas, or complex state machines.
*   **Deep Debugging:** Chasing race conditions, memory leaks, or logical bugs that span multiple unlinked files.
*   **Heavy Refactoring:** Migrating frameworks or restructuring large legacy codebases.
*   **Self-Correction:** Escalation target when Tier 2 repeatedly fails to resolve an issue.

## Routing Evaluation Logic (Pre-Task)
Before executing a task or spawning a sub-agent, evaluate:
1. **Algorithmic Complexity:** Does this require novel logic (Tier 3), standard logic (Tier 2), or no logic (Tier 1)?
2. **Context Window Needs:** If a task requires dumping 50k+ tokens of logs just to find an error, use Tier 1 to filter the logs first, then send the filtered output to Tier 2/3 for analysis.
3. **Determinism:** If the task has a strict right/wrong answer (like fixing a syntax typo), route to Tier 1.

## Escalation Protocol
Do not trap agents in infinite failure loops using cheap models.
1. **Initial Attempt:** Execute at the evaluated tier.
2. **First Failure:** If the test/linter fails, retry ONCE on the same tier with the error log provided.
3. **Escalation:** If the second attempt fails, immediately abort the current tier and escalate the prompt and error history to the next higher tier.
4. **Handoff:** When passing state upward, generate a brief `context.md` summarizing what failed so the frontier model does not repeat the cheaper model's mistakes.