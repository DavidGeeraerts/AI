<div align="center">
<img src="/assets/images/Book-Cover-Banner.jpg" alt="The Book of AI Banner"/>
</div

## Mapping the Artificial Intelligence landscape for agentic engineering.

> See how deep the rabbit hole goes: a practical guide to agentic engineering and building your own stack for agentic systems.

## Objective

> Be a comprehensive, well curated (signal from noise), maintained manual for agentic engineering, and building a practical toolkit for local LLM sovereignty.

## :construction: _This book is currently under heavy construction._ :construction:

### :fountain_pen: _Laying the foundation._ :fountain_pen:

:book: [Read The Book of AI online](https://the-book-of-ai.github.io/source)

- :books: [Chapter Overview](chapter-structure.qmd)

## Start Here (What's Ready Now)

- :calendar: [Monthly digest catalog](https://the-book-of-ai.github.io/source/digest/digest_index.html) — dated snapshots of current tools and ideas.
- :tv: [Daily YouTube digest](https://the-book-of-ai.github.io/youtube-feed-dashboard-ai/) — curated videos about agentic engineering.
- :green_book: [Chapter 02: Models](https://the-book-of-ai.github.io/source/chapters/02-models.html) — _Status: Draft_
- :orange_book: [Chapter 7: Resources](https://the-book-of-ai.github.io/source/chapters/07-resources.html) — _Status: Draft_
- :green_book: [Glossary](https://the-book-of-ai.github.io/source/appendices/glossary.html) — _Status: Draft_
- :green_book: [Prompt Engineering Notes](https://the-book-of-ai.github.io/source/appendices/prompt-engineering-notes.html) — _Status: Review_

---

## :snail: Book Progress

### Current Focus

Looking at [claims](/sources/claims.yml) as a [reference](references.bib) for supporting evidence and examples for the book. The goal is to have a curated set of claims with supporting evidence and examples that can be used to support the arguments in the book.

#### Claims Relationship Diagram

```text
references.bib
  ├── @katz2026agentharness
  └── @ibm2026agenticengineeringterms
         │
         └── cited by
                │
claims.yml
  └── ch04-harness-repeatability
         │
         ├── source: 7Y4sbZgBdnc 18:12–19:38
         └── source: SIZsFGvtClM 24:04–25:11
                │
                └── displayed in
                       │
                       └── chapters/04-agents-and-harnesses.qmd
```

### Status chart

| Chapters                          | Title                    | Status   | Active          |
|-----------------------------------|--------------------------|----------|-----------------|
| Index                             | Main page                | revising | :yellow_circle: |
| Chapter 1                         | Introduction             | notes    | :red_circle:    |
| Chapter 2                         | Models                   | draft    | :green_circle:  |
| Chapter 3                         | Running local models     | draft    | :yellow_circle: |
| Chapter 4                         | Agents and Harnesses     | draft    | :red_circle:    |
| Chapter 5                         | Building a working stack | notes    | :yellow_circle: |
| Chapter 6                         | Miscellaneous Tools      | notes    | :red_circle:    |
| Chapter 7                         | Resources                | review   | :yellow_circle: |
| Appendix-Glossary                 | Glossary                 | revising | :yellow_circle: |
| Appendix-Prompt Engineering Notes | Prompt Engineering Notes | final    | :red_circle:    |
| Appendix-Digest Catalog           | Digest Catalog           | draft    | :yellow_circle: |
| Appendix-Bibliography             | Bibliography             | draft    | :yellow_circle: |

#### Status key

| Status      | Meaning                                      | Description                        |
|-------------|----------------------------------------------|------------------------------------|
| notes       | Scattered notes / outline only               | Pre-writing                        |
| outline     | Structured outline exists                    | Ready to start drafting            |
| rough draft | rough draft written                          | some Content is there but rough    |
| draft       | First full draft written                     | Content is there but needs editing |
| revising    | Actively editing / rewriting                 | Working on it                      |
| review      | Ready for feedback or self-review            | Polishing stage                    |
| final       | Considered done (may still get minor tweaks) | Chapter is solid                   |
| active      | Currently being worked on                    | Highlight current focus            |

## Glossary technical reference

Examples of how to link to glossary terms in the text:

```text
[{{< glossary "(AI) Artificial Intelligence" display="AI" popup="hover" >}}](https://the-book-of-ai.github.io/source/appendices/glossary.html#ai-artificial-intelligence)
```

## Work with the source

The book is built with [Quarto](https://quarto.org/). Its chapters and appendices are Quarto Markdown (`.qmd`) files, and [_quarto.yml](_quarto.yml) defines the reading order and publishing settings.
With Quarto installed, run `quarto render` from the repository root to build the website. The generated site goes to `docs/`; make changes to the source files, not the generated output. Corrections and suggestions are welcome through [GitHub issues](https://github.com/the-book-of-ai/source/issues).

# Resource References

- [Github emoji cheat sheet](https://github.com/ikatyang/emoji-cheat-sheet) -- for using emojis in markdown
- [Quarto website](https://quarto.org/) -- publishing and rendering engine for the book
- [Mermaid](https://mermaid.js.org/) -- create diagrams and visualizations using text and code

# Quotes

> "Not all benchmarks lie, but all liars benchmark"

> "GPU sovereignty is the new second amendment" ~DHH

# History

_This is a work in progress --started 2025-02-- to document [my](https://github.com/DavidGeeraerts) exploration of Artificial Intelligence, Large Language Models (LLM), Natural Language Models (NLM), Natural Language Processing (NLP), agents, agentic engineering, etc._

---

## The Why

| W's                |                                                                                                                  |
|--------------------|------------------------------------------------------------------------------------------------------------------|
| **Who** it’s for:  | Engineers and technical practitioners building and working with AI agents.                                       |
| **What** it is:    | A curated comprehensive, reference and architecture book for agentic systems engineering.                        |
| **Why** it exists: | A single, maintained source of knowledge instead of scattered resources for practitioners of agentic engineering |

---

# License

Unless otherwise noted, original content in this repository by the-book-of-ai is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE). Please attribute it as “The Book of AI" and link to this repository and the license. Third-party materials, including research papers and images, are excluded and remain subject to their own terms.

# 📊 GitHub Stats

![Alt](https://repobeats.axiom.co/api/embed/a45c742cf7979dcff3ac7e99445bc37a05abb2d0.svg "Repobeats analytics image")
