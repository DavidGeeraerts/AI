# The Book of AI - Artificial Intelligence landscape
This is a work in progress --started 2025-02-- to document my exploration of Artificial Intelligence, Large Language Models (LLM), Natural Language Models (NLM), Natural Language Processing (NLP), etc. 



## [Glossory](Glossory.md)
## [Notes](Notes.md)

## :fire: Hotplate

### [OpenWaldo](https://openwaldo.org/)
| Letter | Meaning |
| --- | --- |
| W | Weights | 
| A | Artifacts |
| L | Licenses |
| D | Data |
| O | Origin |

OpenWALDO brings everything people expect from open source to AI: a community anyone can join; source code and training data anyone can contribute to and audit; open tools; public governance and trust earned through transparency; and models anyone can compose, train, validate, reproduce, extend, and improve.

### [Ajax](https://data.pewdiepie.com/)
Ajax is a fine-tuned (abliterated) Qwen 3.5 9B model, trained for Odysseus to be your always on agent. It handles your daily tasks from search to browse web to email to your calendar - all your daily tasks completely privately. Ajax's refusal has been ablated for a freer, less restricted AI experience. Please use responsibly.

### [Odysseus](https://odysseusai.dev/)
Odysseus is a self-hosted AI workspace for chat, agents, deep research, email, calendar, and local/API model backends. It is local-first and privacy-first when you run local models on your own machine.

### [BenchLM](https://benchlm.ai/)
LLM Leaderboard.
#### [Best Ollama models](https://benchlm.ai/best/ollama-models)

### [Heretic](https://heretic-project.org/)
Heretic removes restrictions (abliteration parameters) from language models, making sure they always follow your instructions.



## AI Model Types
- Large Language Models (LLM)
  - General-purpose language models
  - Code-specialized models {GitHub Copilot (Codex), Anthropic Claude Code, StarCoder, CodeLlama, DeepSeek Coder}
- Image Generation Models
  - Text-to-Image {Stable Diffusion, Midjourney, DALL-E}
  - Image-to-Image
- Multimodal Models (Text, Image, Audio, Video) {OpenAI Sora, Google Gemini, Meta Seamless, }

### Prominent AI Organizations Creating Foundational AI Models
(Order by Name ↑)

- [Alibaba](https://www.alibabacloud.com/en)
- [Allen Institute for AI](https://allenai.org/)
- [Amazon AI](https://aws.amazon.com/machine-learning/)
- [Anthropic](https://www.anthropic.com/)
- [xAI](https://x.ai/)
- [Cohere](https://cohere.ai/)
- [Deepseek](https://www.deepseek.com/)
- [Google AI](https://ai.google/)
- [Hugging Face](https://huggingface.co/)
- [IBM](https://www.ibm.com/granite)
- [Meta AI](https://ai.facebook.com/)
- [Microsoft AI](https://www.microsoft.com/en-us/ai)
- [Mistral](https://mistral.ai/en)
- [OpenAI](https://openai.com/)
- [Stability AI](https://stability.ai/)

### Foundational AI Models
(Order by Name ↑)
- [Claude](https://www.anthropic.com/claude)
- [Deepseek](https://www.deepseek.com/)
- [GPT](https://openai.com/index/gpt-4/)
- [Granite](https://www.ibm.com/granite)
- [Llama](https://www.llama.com/)
- [Nova](https://aws.amazon.com/ai/generative-ai/nova/)
- [o1](https://openai.com/o1/)
- [Qwen](https://huggingface.co/Qwen)
- [Stable Diffusion](https://stability.ai/stable-image)

### Base Model Data
(Order by Name ↑)
- [AWS Open Data](https://aws.amazon.com/marketplace/search/results?trk=8384929b-0eb1-4af3-8996-07aa409646bc&sc_channel=el&FULFILLMENT_OPTION_TYPE=DATA_EXCHANGE&CONTRACT_TYPE=OPEN_DATA_LICENSES&filters=FULFILLMENT_OPTION_TYPE%2CCONTRACT_TYPE)
- [**Common Crawl**](https://commoncrawl.org/)
- [**FineWeb**](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) -- curated Internet dataset.
- [**Google Dataset Search**](https://datasetsearch.research.google.com/)
- [**Hugging Face Datasets**](https://huggingface.co/datasets)
- [**Kaggle**](https://www.kaggle.com/datasets)
- [Microsoft Planetary Computer Data Catalog](https://planetarycomputer.microsoft.com/catalog)
- [MNIST (Modified National Institute of Standards and Technology database)](https://github.com/cvdfoundation/mnist)
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
- [SNAP - Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/)
- [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/)
- [USA Data Gov](https://data.gov/)

### Miscellaneous Tools
(Order by Name ↑)
- [Excalidraw](https://excalidraw.com/) - online Diagram tool
- [LLM Visualization](https://bbycroft.net/llm)
- [Tiktokenizer](https://tiktokenizer.vercel.app/)

### Benchmarks
| Benchmark | Status in 2026 | Strengths | Weaknesses | Still useful? |
|-----------|----------------|-----------|------------|---------------|
| **MMLU** | Saturated | Historical reference for broad knowledge | Frontier models cluster ~90%+; little differentiation; contamination risk | Mostly no (use MMLU-Pro if needed) |
| **GSM8K** | Saturated | Simple math word problems | Top models near ceiling (~95%+); many invalid questions noted | No for frontier comparison |
| **GPQA** (esp. Diamond) | Near-saturating | Graduate-level “Google-proof” science reasoning | Leaders pushing 90%+; small set → noise | Still decent for science reasoning, but headroom shrinking |
| **SWE-bench** (Verified/Pro variants) | Contested / partially saturating | Real GitHub issue resolution; strong signal for coding agents | Contamination concerns; scores climbing into high 70s–90s on easier versions | Yes for coding/agentic work (prefer newer/harder variants like Pro) |
| **GAIA** | Active but contested | Multi-step agentic tasks (browsing, tools, files) that are easy for humans | Large score gaps depending on scaffolding/tools; some saturation reports | Yes for assistant/agent evaluation |
| [**LiveBench**](https://livebench.ai/#/) | Highly regarded | Monthly-refreshed questions from recent sources; objective ground-truth scoring; covers reasoning, coding, math, data analysis, language, instruction following | Not human preference; not pure agentic long-horizon | **One of the strongest current objective capability benchmarks** |
| [**LMArena**](https://lmarena.ai/) (Chatbot/Arena) | Highly regarded | Large-scale blind human preference (Elo-style); real user prompts | Style/verbosity bias; overlapping confidence intervals; not verifiable correctness | **Best for “which model feels best in open-ended use”** |
| [**BenchLM**](https://benchlm.ai/) | Aggregator | Combines many sources into overall indices | Depends on underlying benchmarks; not a primary eval itself | Useful meta-view, not a standalone benchmark |
| **Mensa Norway IQ Test** | Niche | Attempts IQ-style scoring | Narrow, not standard in frontier AI evaluation | Rarely used for serious model comparison |
| **WebDev Arena** | Specialized | Web development tasks | Narrow domain | Useful only if you care specifically about web-dev agents |


###  Running AI Models Locally

- [ollama](https://ollama.com) - CLI
- [ollama-ui](https://ollama-ui.github.io/ollama-ui/) - Simple HTML UI for Ollama. Available as [Chrome extension](https://chromewebstore.google.com/detail/ollama-ui/cmgdpmlhgjhoadnonobjeekmfcehffco?pli=1).
- [LM Studio](https://lmstudio.ai/) - GUI

| Aspect              | **LM Studio**                                      | **Ollama**                                          |
|---------------------|----------------------------------------------------|-----------------------------------------------------|
| **Primary focus**   | Polished desktop GUI for discovery, chat & experimentation | CLI + background service for scripting, APIs & integrations |
| **License**         | Proprietary (free for personal + internal business use) | Fully open-source (MIT)                             |
| **Interface**       | Excellent GUI + CLI (`lms`) + headless daemon (`llmster`) | CLI-first + local API; has added a basic desktop app |
| **Model discovery** | Built-in Hugging Face browser with size/VRAM estimates | Curated library + simple `ollama pull model:tag`    |
| **API**             | OpenAI- + Anthropic-compatible (port 1234)         | OpenAI-compatible + native endpoints (port 11434)   |
| **Best hardware edge** | Stronger MLX performance on Apple Silicon         | Lower overhead, often slightly faster on NVIDIA    |
| **Headless/server** | Yes (llmster since early 2026)                     | Native from the start (systemd/Docker-friendly)     |
| **Extras**          | Built-in document chat (RAG), MCP client, LM Link (remote device sharing), continuous batching | Excellent ecosystem integrations, Modelfiles, official Docker image, optional cloud models |


#### When Ollama is usually betterYou want a fully open-source, auditable tool.
- You are a developer building scripts, agents, or apps (many frameworks and coding tools assume Ollama by default).
- You need Docker, Kubernetes, or long-running headless servers with minimal footprint.
- You prefer the lowest idle RAM and fastest cold-start times (especially on NVIDIA GPUs).
- You want the absolute simplest CLI workflow (ollama run model).


## Web/Online Models
(Order by Name ↑)
- [Allen Institute AI - Tulu 3:405B](https://playground.allenai.org/)
- [Anthropic](https://www.anthropic.com/) - [Claude](https://claude.ai/new)
- [Deepseek](https://www.deepseek.com/) - [R1](https://chat.deepseek.com)
- [GitHub Models](https://github.com/marketplace/models)
- [Google AI Studio](https://aistudio.google.com/prompts/new_chat)
- [Granite](https://www.ibm.com/granite/playground/)
- [Grok](https://grok.com/)
- [Nova](https://chat.novaapp.ai/)
- [Qwen](https://chat.qwenlm.ai/)
- [Together AI](https://www.together.ai/)


## AI [blog] Resources
(Order by Name ↑)

- [AI News](https://buttondown.com/ainews) - [RSS](https://buttondown.com/ainews/rss)
- [Ai2-Allen Institute](https://allenai.org/blog)
- [ApX Machine Learning](https://apxml.com/posts)
- [Arsturn](https://www.arsturn.com/blog)
- [Epoch AI](https://epoch.ai/)
- [ShinChven's Blog](https://atlassc.net/)


## Prominent AI People
(Sorted by Surname ↑)

- [Dario Amodei](https://darioamodei.com/) - [Anthropic](https://www.anthropic.com/)
- [AmandA Askell](https://askell.io/) - [Anthropic](https://www.anthropic.com/)
- [Lex Fridman](https://lexfridman.com/) - [MIT](https://www.mit.edu/)
- [Salim Ismail](https://salimismail.com/) - [Anthropic](https://www.anthropic.com/)
- [Nathan Lambert](https://www.interconnects.ai/) - [Allen Institute for AI](https://allenai.org/)
- [Emad Mostaque](https://emad.posthaven.com/) - [stability.ai](https://stability.ai/)
- [Christopher Olah](https://colah.github.io/) - [Anthropic](https://www.anthropic.com/)


## AI Publication Websites
(Order by Name ↑)

- [arxiv](https://arxiv.org/list/cs.AI/recent)


## AI Publications
(Sorted by Publication Date ↓)

- 2025-04-25  [BitNet b1.58 2B4T Technical Report](https://arxiv.org/pdf/2504.12285)
- 2024-04-09  [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654)
- 2023-07-27  [Universal and Transferable Adversarial Attacks on Aligned Language Models](./Publications/Universal%20and%20Transferable%20Adversarial%20Attacks%20on%20Aligned%20Language%20Models.pdf) - [LLM Attacks](https://llm-attacks.org/)
- 2019-06-17  [Superposition of many models into one](https://arxiv.org/abs/1902.05522)
- 2017-06-12  [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Articles
(Sorted by Publication Date ↓)
- 2025-02-12 [Unlocking the Effective Context Length: Benchmarking the Granite-3.1-8b Model](https://www.redhat.com/en/blog/unlocking-effective-context-length-benchmarking-granite-31-8b-model)
- 2025-02-02 [A Detailed Analysis of Fine-Tuning, Direct Preference Optimization (DPO), and Reinforcement Learning with Verifiable Rewards (RLVR) on the LLama3.1 405B Model](https://medium.com/@zhouboyang1983/a-detailed-analysis-of-fine-tuning-direct-preference-optimization-dpo-and-reinforcement-c24d9061cd84)
- 2025-01-31 [DeepSeek-V3 Explained 1: Multi-head Latent Attention](https://towardsdatascience.com/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4/)
- 2025-01-19  [Top 5 Mistakes to Avoid When Learning Machine Learning](https://apxml.com/posts/top-mistakes-when-learning-machine-learning)
- 2024-10-02  [How to Get Started with Machine Learning: A Beginner’s Step-by-Step Guide](https://apxml.com/posts/get-started-with-machine-learning-guide)
- 2024-07-13  [MHA vs MQA vs GQA vs MLA](https://medium.com/@zaiinn440/mha-vs-mqa-vs-gqa-vs-mla-c6cf8285bbec)
- 2020-00-00  [Over 200 of the Best Machine Learning, NLP, and Python Tutorials — 2018 Edition](https://robbieallen.medium.com/over-200-of-the-best-machine-learning-nlp-and-python-tutorials-2018-edition-dd8cf53cb7dc)

## Online Books
(Sorted by Publication Date ↓)
- 2024  [Machine Learning and Deep Learning with R](https://theoreticalecology.github.io/machinelearning/)
- 2024  [Natural Language Processing (NLP) Course](https://srdas.github.io/NLPBook/intro.html)
- 2023  [State of Open Source AI Book - 2023 Edition](https://book.premai.io/state-of-open-source-ai/)
- 2023  [Understanding Deep Learning](https://udlbook.github.io/udlbook/)
- 2019  [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- 2018  [Deep Learning](https://srdas.github.io/DLBook/)
- 2016  [Deep Learning](https://www.deeplearningbook.org/)

## YouTube Videos & Channels
(Sorted by Publication Date ↓)
- 2026-05-13  [The Coding Gopher: Uni-1 AI image generation model](https://youtu.be/xhaE2sUnUZE?si=otuy7ytKM3nXzbFn) 
- 2026-03-26  [The Diary of a CEO: AI Whistleblower: We Are Being Gaslit By AI Companies, They’re Hiding The Truth! - Karen Hao](https://youtu.be/Cn8HBj8QAbk?si=1CGH8ELHuUwIP39-)
- 2025-04-28  [Logically Answered: AMD's $243 Billion AI Disaster...What Happened?](https://youtu.be/SDIq26_DVLk?si=RqlxfKt6Gff-IG4M)
- 2025-04022  [Computerphile: What is Cuda?](https://youtu.be/K9anz4aB0S0?si=VO5Uy9iJ5AKajEnP)
- 2025-03-05  [ByteByteGo: What Is the Most Popular Open-Source AI Stack?](https://www.youtube.com/watch?v=hFURlsMwU7c)
- 2025-02-05  [Andrej Karpathy: Deep Dive into LLMs like ChatGPT](https://youtu.be/7xTGNNLPyMI?si=u2q8zaBubzTCPKC7)
- 2025-02-02  [Lex Fridman: DeepSeek, China, OpenAI, NVIDIA, xAI, TSMC, Stargate, and AI Megaclusters | Lex Fridman Podcast #459](https://youtu.be/_1f-o0nqpEI?si=b3xO1D6jP-5g06e8)
- 2025-01-29  [Peter H. Diamandis: DeepSeek vs. Open AI - The State of AI w/ Emad Mostaque & Salim Ismail | EP #146](https://youtu.be/lY8Ja00PCQM?si=71XmR5B_VikMcYCg)
- 2024-11-11  [Lex Fridman: Dario Amodei: Anthropic CEO on Claude, AGI & the Future of AI & Humanity | Lex Fridman Podcast #452](https://youtu.be/ugvHCXCOmm4?si=QekDqk5yuNA5fR1H)
- 2024-09-09  [IBM Technology: RAG vs. Fine Tuning](https://youtu.be/00Q0G84kq3M?si=iJn3KAoFzOMM8a-L)
- 2024-04-01  [3Blue1Brown: Neural networks Playlist](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=xFClf3TeCbAQkgok)
- 2023-08-23  [IBM Technology: What is Retrieval-Augmented Generation (RAG)?](https://youtu.be/T-D1OfcDW1M?si=Uv-M2QcZL7WIJi40)
- 2018-07-30  [Lex Fridman: Deep Learning State of the Art (2020)](https://youtu.be/0VH1Lim8gL8?si=84lMRmcAPNYN6Lm-)
- [Channel: StatQuest with Josh Starmer](https://www.youtube.com/@statquest)
