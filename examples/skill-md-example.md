---
name: skill-name-here
description: One-sentence description of what this skill does and when to use it.
version: 1.0.0
author: Your Name or Handle
license: MIT
triggers:
  - keyword or phrase that should activate this skill
  - another trigger
tags:
  - category1
  - category2
---

# Skill Name

Short introduction (1–3 sentences) explaining the purpose of this skill and the problem it solves.

## When to use this skill

- Clear bullet list of situations where this skill should be activated
- Another good use case
- Avoid using it when…

## Instructions

You are now operating under this skill. Follow these rules strictly:

1. **First step** – Always do this first.
2. **Core process** – Detailed, numbered steps the agent must follow.
3. **Verification** – How to check that the work is correct before finishing.
4. **Output format** – Exactly what the agent should return or write.

### Detailed Guidelines

- Prefer deterministic tools and code over pure model judgment when possible.
- Keep changes minimal and focused.
- Always run tests / checks after making changes.
- If blocked, clearly state the blocker and what is needed.

## Examples

### Example 1: Basic usage

**Input:**  
Brief description of a typical request.

**Expected behavior:**  
What the agent should do step by step.

### Example 2: Edge case

**Input:**  
A more difficult or edge-case request.

**Expected behavior:**  
How the skill should handle it.

## Constraints & Guardrails

- Never do X
- Always require approval before Y
- Do not modify files outside of Z
- Maximum number of iterations / tool calls

## Related Skills

- `related-skill-1` – when to hand off
- `related-skill-2`

## Notes

Any additional context, version history, or maintenance notes.