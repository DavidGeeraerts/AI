# Models

Published

2026-10-27

Choosing a model starts with understanding its capabilities, availability, and trade-offs.

## Model categories

- General-purpose and code-specialized large language models
- Multimodal models
- Image-generation models, including text-to-image and image-to-image systems
- Decision or “System One” models, including calibrated-decision approaches

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

## Model families and access

The landscape includes models from Anthropic (Claude), OpenAI (GPT), Google DeepMind (Gemini and Gemma), DeepSeek, Alibaba (Qwen), Meta (Llama and Muse), xAI (Grok), Moonshot (Kimi), Mistral, and Zhipu (GLM). Open-weight and closed models differ in how they can be inspected, hosted, adapted, and accessed.

## Evaluation and training data

Benchmarks help compare models, but scores need context. Some benchmarks, such as MMLU and GSM8K, are nearing saturation; others, including SWE-bench variants, LiveBench, LMArena, GAIA, and BenchLM, provide different signals. Training data sources such as Common Crawl and FineWeb also shape model capabilities.

## Abliteration

Refusal-removal techniques and fine-tunes, including Heretic and Ajax, can change how a model responds. Consider their impact on behavior and agent autonomy when selecting a model.

## Prominent AI Organizations Creating Foundational AI Models

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

## Benchmarks

| Benchmark | Status in 2026 | Strengths | Weaknesses | Still useful? |
|----|----|----|----|----|
| **MMLU** | Saturated | Historical reference for broad knowledge | Frontier models cluster ~90%+; little differentiation; contamination risk | Mostly no (use MMLU-Pro if needed) |
| **GSM8K** | Saturated | Simple math word problems | Top models near ceiling (~95%+); many invalid questions noted | No for frontier comparison |
| **GPQA** (Graduate-Level Google-Proof Q&A Benchmark; esp. Diamond) | Near-saturating | Challenging multiple-choice benchmark evaluating advanced scientific reasoning and deep domain understanding in large language models | Leaders pushing 90%+; small set → noise | Still decent for science reasoning, but headroom shrinking |
| **SWE-bench** (Verified/Pro variants) | Contested / partially saturating | Real GitHub issue resolution; strong signal for coding agents | Contamination concerns; scores climbing into high 70s–90s on easier versions | Yes for coding/agentic work (prefer newer/harder variants like Pro) |
| **GAIA** | Active but contested | Multi-step agentic tasks (browsing, tools, files) that are easy for humans | Large score gaps depending on scaffolding/tools; some saturation reports | Yes for assistant/agent evaluation |
| [**LiveBench**](https://livebench.ai/#/) | Highly regarded | Monthly-refreshed questions from recent sources; objective ground-truth scoring; covers reasoning, coding, math, data analysis, language, instruction following | Not human preference; not pure agentic long-horizon | **One of the strongest current objective capability benchmarks** |
| [**LMArena**](https://lmarena.ai/) (Chatbot/Arena) | Highly regarded | Large-scale blind human preference (Elo-style); real user prompts | Style/verbosity bias; overlapping confidence intervals; not verifiable correctness | **Best for “which model feels best in open-ended use”** |
| [**BenchLM**](https://benchlm.ai/) | Aggregator | Combines many sources into overall indices | Depends on underlying benchmarks; not a primary eval itself | Useful meta-view, not a standalone benchmark |
| **Mensa Norway IQ Test** | Niche | Attempts IQ-style scoring | Narrow, not standard in frontier AI evaluation | Rarely used for serious model comparison |
| **WebDev Arena** | Specialized | Web development tasks | Narrow domain | Useful only if you care specifically about web-dev agents |
| [**Humanity’s Last Exam**](https://lastexam.ai/) (HLE) | Active frontier benchmark | Expert-authored (≈1,000 contributors) multi-modal closed-ended questions (2,500 total) at the frontier of human knowledge across 100+ subjects; designed to resist search/contamination; objective grading; HLE-Diamond & Rolling variants for continued headroom | Score variance across full set / text-only / Diamond / with-tools; some answer-key disputes (esp. chem/bio); calibration remains poor | **Yes — currently one of the strongest non-saturated academic/knowledge + reasoning benchmarks** |
| [**WorkBuddy Bench**](https://github.com/Tencent/workbuddy-bench) (Tencent) | Active / Research Preview (v1.2, Oct 2026) | Multi-domain coding-agent eval (Code 80 / Web 70 / Office 50 / Security 60 tasks) reverse-engineered from real work; contamination-resistant prompts; dual harnesses (CodeBuddy Code + Claude Code); fully open (tasks, Docker images, tests, solutions); reports efficiency (tokens/turns) | No suite-wide average (subset scores incomparable); harness effects are large; still early / Research Preview; heavier compute than pure unit-test benchmarks | **Yes — one of the strongest current multi-domain agentic coding benchmarks**, especially for real-world role-played work beyond pure SWE-bench-style issue fixing |

### Benchmark Articles

- [Grading the Graders: Gaming Benchmarks and Making Your Own](https://interestingengineering.substack.com/p/grading-the-graders-gaming-benchmarks)

## Base Model Data

(Order by Name ↑)

- [AWS Open Data](https://aws.amazon.com/marketplace/search/results?trk=8384929b-0eb1-4af3-8996-07aa409646bc&sc_channel=el&FULFILLMENT_OPTION_TYPE=DATA_EXCHANGE&CONTRACT_TYPE=OPEN_DATA_LICENSES&filters=FULFILLMENT_OPTION_TYPE%2CCONTRACT_TYPE)
- [**Common Crawl**](https://commoncrawl.org/)
- [**FineWeb**](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) – curated Internet dataset.
- [**Google Dataset Search**](https://datasetsearch.research.google.com/)
- [**Hugging Face Datasets**](https://huggingface.co/datasets)
- [**Kaggle**](https://www.kaggle.com/datasets)
- [Microsoft Planetary Computer Data Catalog](https://planetarycomputer.microsoft.com/catalog)
- [MNIST (Modified National Institute of Standards and Technology database)](https://github.com/cvdfoundation/mnist)
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
- [SNAP - Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/)
- [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/)
- [USA Data Gov](https://data.gov/)

Back to top
