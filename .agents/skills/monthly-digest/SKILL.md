---
name: monthly-digest
description: Research, write, and integrate a monthly AI digest for this project.
---

# Monthly Digest

Use this skill when asked to research, create, revise, or publish an issue of the
project's monthly AI digest.

## Project context

- Digest issues live in `digest/YYYY-MM-Month-digest.qmd`; use the month’s full
  English name, as in `digest/2026-11-November-digest.qmd`.
- `digest/digest_index.qmd` contains the digest catalog. Keep it newest-first:
  `scripts/latest-digest.ts` uses its first entry to update the homepage link.
- `_quarto.yml` lists the current digest as a book chapter. Keep that list in
  sync when adding a new issue.
- `appendices/SOURCES.qmd` is the curated source directory and the starting point
  for web research. It is an index, not a substitute for checking each source.
- Read the root `AGENTS.md` and the newest issue before editing. Follow its
  repository-wide constraints.

## Workflow

1. Establish the requested month and year. If they are unclear and would change
   the issue being edited, ask before proceeding. Check the working tree and
   preserve unrelated user changes.
2. Inspect the latest digest or two and the catalog to match the project's
   Quarto/Markdown conventions without assuming every issue needs identical
   sections.
3. Research developments from the issue's month, starting with relevant entries
   in `appendices/SOURCES.qmd`. Prefer original announcements, papers, release
   notes, and benchmark documentation; use credible reporting for context.
   Verify dates, links, names, and quantitative claims against the source.
4. Select items for their practical relevance to agentic engineering: agents and
   harnesses, models, tools, evaluations, infrastructure, workflows, and
   important implications for builders. Favor meaningful developments over a
   long list of minor launches.
5. Write the issue in `digest/YYYY-MM-Month-digest.qmd`. Use a clear period
   heading and a concise opening synthesis, followed by descriptive topic
   sections. Give each item a linked title and a compact, original explanation
   of what it does or why it matters. Use comparison tables when they clarify
   real trade-offs; distinguish reported facts from analysis and qualify
   uncertainty. Do not invent details or copy source prose.
6. When adding an issue, add it to the top of the digest catalog and add its
   path to the book chapter list in `_quarto.yml`, without duplicating entries.
   Do not manually update the homepage's “Read the latest Digest” link; the
   pre-render script manages it from the catalog.
7. Render the Quarto project as required by `AGENTS.md`, then inspect the output
   for rendering errors and verify that the new issue appears in the book and
   that the catalog points to the newest issue.

## Boundaries

- Do not edit `docs/`, `_book/`, `.quarto/`, `_extensions/`, or
  `appendices/glossary.yml`. Do not modify unrelated book chapters or source
  materials as part of digest work.
- If rendering would change protected generated output, do not edit that output
  by hand; report the conflict and use a safe render destination if supported.
- Keep claims linked to sources, and omit or clearly label claims that cannot be
  independently verified.

## Completion

Report the issue created or updated, any catalog/configuration files changed,
the research limitations (if any), and the render result. Do not claim a render
passed unless it completed successfully.
