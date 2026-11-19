work in progress

## `CONTRIBUTING.md`

Place this at the root of the repository so GitHub surfaces it automatically on issues and pull requests.

```markdown
# Contributing to *The Book of AI*

A reference book only stays useful if people maintain it. Corrections, updated benchmarks, clearer diagrams, and new lab modules are welcome.

## Repository Layout

* `chapters/` — Core book chapters (e.g., `05-building-a-working-stack.qmd`).
* `appendices/` — Reference material, including `glossary.qmd` and `colophon.qmd`.
* `_quarto.yml` — Book structure, navigation, and render settings.
* `.markdownlint.jsonc` — Active Markdown linting rules.

Reusable microlabs and shared snippets use Quarto `{{< include >}}` shortcodes so they can be maintained in one place.

## Local Setup

1. Fork and clone the repository:
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/AI.git
   cd AI
