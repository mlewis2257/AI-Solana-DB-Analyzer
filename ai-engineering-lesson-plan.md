# AI Engineering — Weekly Lesson Plan

_Instructor: Structure mirrors the ByteByteAI 6-project arc. Each week = subject matter breakdown + curated resources + project spec + checkpoint. Stack choices lean on what you already know: Python for model/data work, Node/Express + JS for backend/real-time, your Postgres VPS instead of new infra._

---

## WEEK 1 — LLM Foundations: Tokenizers & a Tiny GPT

### Subject matter (do in this order)

1. **Tokenization (BPE)** — why it exists, why it breaks arithmetic/spelling/non-English text
2. **Transformer architecture** — embeddings, positional encoding, self-attention, multi-head attention, feedforward blocks, layernorm, residual connections
3. **Training loop** — next-token prediction, cross-entropy loss, train/val split
4. **Sampling strategies** — greedy, temperature, top-k, top-p/nucleus, beam search
5. **Post-training overview (conceptual only this week)** — SFT vs. RLHF, why base models "dream text" but chat models "answer"

### Resources

- **Video series:** Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out" (YouTube) — the core lecture for this week
- **Video:** Karpathy, "Let's build the GPT Tokenizer" (YouTube) — build BPE by hand
- **Text version:** fast.ai's chapter-format writeup of the tokenizer video (fast.ai blog, Oct 2025) — good for re-reading without re-watching 2h13m of video
- **Repo to fork:** `karpathy/build-nanogpt` (GitHub) — reference implementation, use as a check against your own code, not a copy-paste
- **Repo (lighter weight):** `karpathy/nanoGPT` (GitHub) — the original minimal trainer; good for the Shakespeare char-level run (~3 min on a GPU, or slower on CPU/your homelab hardware)
- **Interactive tool:** tiktokenizer web app — paste text, watch tokenization live in-browser
- **If you want the $100 "own ChatGPT" stretch goal later:** Karpathy's `nanochat` repo (Oct 2025) — full pipeline including SFT/RL, more advanced than this week needs but worth bookmarking for Week 2-3 crossover

### Build

- Implement BPE tokenizer from scratch in Python (no library) — train it on a text corpus of your choice, inspect the merge rules
- Train a char-level tiny GPT on Tiny Shakespeare (nanoGPT config) — get it running end-to-end on your own hardware or the VPS
- Build a small CLI or Streamlit playground (reuse your Data Summarizer Streamlit muscle memory) that lets you flip between greedy / top-k / top-p sampling on the same prompt and see the difference

### Checkpoint (end of week)

You should be able to explain, without notes: why GPT-2 can't reverse a string, why token counts don't match word counts, and what changes in the generated text when you move from greedy to top-p sampling.

---

## WEEK 2 — RAG: COLOR Support Chatbot

### Subject matter

1. **Adaptation methods overview** — full fine-tuning vs. PEFT/LoRA, and _why RAG beats fine-tuning for this use case_ (freshness, no retraining, auditability)
2. **Document processing** — parsing, chunking strategies (fixed-size, recursive character, semantic/token-aware), overlap tradeoffs
3. **Embeddings & indexing** — embedding models, HNSW vs. IVF/IVFFlat indexes, why pgvector fits your stack specifically
4. **Retrieval** — dense vs. keyword (BM25) vs. hybrid, multi-query retrieval, reranking (cross-encoder or LLM-based)
5. **Generation** — prompt construction with retrieved context, citation-aware answers
6. **Evaluation** — RAGAS's four metrics: context precision, context recall, faithfulness, answer relevancy

### Resources

- **Primary course:** DeepLearning.AI, "Building and Evaluating Advanced RAG" (Jerry Liu, LlamaIndex + TruEra) — ~4-5 hrs, focuses specifically on evaluation, which most tutorials skip
- **Vendor-authoritative:** LangChain Academy (official, self-paced) — RAG + the newer LangGraph module
- **Hands-on walkthrough:** "Building Your Own RAG" (Medium/CodeToDeploy, May 2026) — full pipeline: load → chunk → embed → store → retrieve → generate, using Chroma for local dev before you move to pgvector
- **pgvector-specific:** search "pgvector RAG pipeline tutorial" once you're ready to wire it into Postgres — since you already run Postgres on your Hostinger VPS, this is the natural production target, no new infra needed
- **Evaluation framework:** RAGAS docs + `ragas` Python package — note current versions may differ from older tutorials' syntax, check the docs directly before copying code
- **Optional deep dive:** ActiveLoop's free RAG course (built with LlamaIndex) if you want the comparative LangChain-vs-LlamaIndex tour

### Build

- Ingest COLOR's docs, venue policies, FAQ, and Trello/sprint notes into a vector store — **use Postgres + pgvector on your existing VPS**, not new infra
- Implement chunking (start with recursive character splitting, ~800 chars/100 overlap, then experiment)
- Build retrieval (start dense-only, then add hybrid BM25 + vector if time allows) and a generation step that cites which chunk it pulled from
- Build a golden set of 30-50 Q/A pairs about COLOR and run RAGAS against your pipeline — this is the step most people skip; don't skip it
- Stretch: wrap it as a real internal tool — a Slack bot or simple web UI the team could actually use

### Checkpoint

You should have a working RAG bot answering real questions about COLOR with citations, plus a RAGAS score sheet showing faithfulness/relevancy numbers before and after at least one tuning change (e.g., chunk size, reranking).

---

## WEEK 3 — Tool-Calling Agent (Ask-the-Web style)

### Subject matter

1. **Workflows vs. agents** — Anthropic's framing: workflows for predictable multi-step tasks, agents for open-ended model-driven decisions
2. **Composable workflow patterns** — prompt chaining, routing, parallelization, orchestrator-worker, evaluator-optimizer (know when to use each, not just what they are)
3. **Tool calling & MCP** — schema design, why tool _ergonomics_ matter (agents fail by calling the wrong tool, right tool/wrong params, or too few tools), MCP as the standardized way to expose tools
4. **The ReAct loop** — reason → act → observe → repeat, and where it breaks down
5. **Agent evaluation** — this is the hard, often-skipped part: build a real eval set, not vibes

### Resources

- **Primary reading:** Anthropic, "Building Effective AI Agents" (resources.anthropic.com) — read this fully before writing any agent code; it's the clearest framing of workflows-vs-agents and the composable patterns
- **Primary reading:** Anthropic, "Writing effective tools for AI agents" (anthropic.com/engineering) — how to design tool schemas agents can actually use well, plus how to build a tool-use eval
- **Advanced/optional:** Anthropic, "Code execution with MCP" (anthropic.com/engineering) — once you have a basic agent working, this covers a more token-efficient pattern (agent writes code that calls tools rather than calling tools directly) — genuinely relevant given you'll likely hit context/cost limits fast
- **Framework:** LangGraph docs + `langchain-mcp-adapters` — official bridge from MCP tool schemas to LangChain-compatible tools; use `create_react_agent` as your starting scaffold rather than building the ReAct loop by hand first time
- **MCP spec:** modelcontextprotocol.io docs — read the spec directly rather than only tutorials, since third-party guides go stale fast

### Build

- Backend in **Node/Express** (your comfort zone) hosting a tool-calling agent
- Give it 3 tools to start: web search, a calculator, and a custom "query COLOR's DB" tool — write the tool descriptions carefully (this is the actual skill Anthropic's tools post is about)
- Stream agent reasoning steps to a simple frontend in real time — this is a direct reuse of your `ws_monitor.py`-style WebSocket experience, just in JS this time (JS for WebSocket integrations, per your own rule)
- Build a 15-20 task eval set (some COLOR-specific, some general) and score tool-selection accuracy, not just final-answer correctness

### Checkpoint

A working agent that picks the right tool for a given question at least 80%+ of the time on your eval set, with visible step-by-step reasoning streamed live.

---

## WEEK 4 — Deep Research Agent (Reasoning Models + Inference-Time Scaling)

### Subject matter

1. **Reasoning/"thinking" models** — what o1/o3, DeepSeek-R1, and similar models actually do differently at inference time
2. **Inference-time scaling techniques** — chain-of-thought prompting, parallel sampling + majority vote, sequential self-refinement, tree-of-thought / search-against-a-verifier, Monte Carlo Tree Search (conceptually — R1 explicitly moved away from MCTS in favor of RL, worth knowing why)
3. **Training-time techniques (conceptual)** — R1's actual pipeline: cold-start SFT data → reasoning-focused RL → rejection sampling + SFT → RL across all scenarios. Know this pipeline; it's the cleanest real-world case study of "how do you actually train a reasoning model"
4. **Reward modeling** — outcome reward models (ORM) vs. process reward models (PRM), and why rule-based/verifiable rewards (RLVR) mattered for R1 specifically
5. **Local deployment tradeoffs** — quantization, cost/latency vs. API calls — directly relevant to your homelab/VPS hardware

### Resources

- **Primary paper:** "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (arXiv 2501.12948) — read the actual paper this week, not just summaries; it's unusually readable for a frontier lab paper
- **Companion paper reading:** the Nature-published version of the R1 paper (peer-reviewed, slightly more polished exposition of the same pipeline)
- **Background context (optional but useful):** any writeup comparing o1's CoT-length scaling to R1's RL-only approach — helps you see there are genuinely two different lineages (proprietary CoT-scaling vs. open RL-only) converging on similar results
- **Practical distillation angle:** Red Hat Developer's "Lessons on reproducing R1-like reasoning in small LLMs" — useful if you want to try distilling reasoning traces into a small model on your own hardware rather than training RL from scratch (RL from scratch is not a weekend project; distillation from traces is)

### Build

- Take Week 3's agent and bolt on a reflection/verifier loop: plan sub-questions → search → synthesize → self-critique → revise before final answer
- Deploy a small open reasoning-capable model locally (VPS or homelab) and benchmark cost/latency/quality against calling a hosted reasoning model API for the same task set — write up the tradeoffs, this is a genuinely useful exercise for your systems background
- Optional: try the distillation angle from the Red Hat piece — fine-tune a small model on a handful of reasoning traces and see how much of the "thinking" behavior transfers

### Checkpoint

A written (even brief) comparison: your deep-research agent's answer quality and cost/latency with vs. without the reflection loop, and local vs. API reasoning model, backed by actual numbers from your eval set.

---

## WEEK 5 — Multi-modal Generation (Diffusion Models)

### Subject matter

1. **Generative model landscape** — VAEs, GANs, autoregressive image models, diffusion — high-level tradeoffs (why diffusion won for images/video)
2. **Forward & reverse diffusion process** — adding noise, learning to predict/remove it, the DDPM formulation
3. **Architecture evolution** — UNet (original) → DiT (Diffusion Transformer) — know why the field is shifting toward transformer-based diffusion backbones
4. **Conditioning** — how text-to-image models condition generation on a prompt (cross-attention, CLIP-style text encoders)
5. **Evaluation** — FID, Inception Score, CLIP score — and their limitations
6. **Text-to-video (conceptual)** — what changes: temporal consistency, latent compression networks, why it's dramatically more expensive to train

### Resources

- **Primary course:** Hugging Face Diffusion Models Course (huggingface.co/learn/diffusion-course) — 4 units, free, notebook-based:
  - Unit 1: intro + from-scratch implementation ("Diffusion Models from Scratch" notebook — build the forward/reverse process yourself in PyTorch before touching the `diffusers` library abstractions)
  - Unit 2: fine-tuning + guidance
  - Unit 3: Stable Diffusion (text-conditioned latent diffusion)
  - Unit 4: DiT and beyond — "Scalable Diffusion Models with Transformers" paper is referenced directly here
- **Hands-on tutorial:** Hugging Face `diffusers` docs, "Train a diffusion model" — trains a UNet2DModel from scratch on the Smithsonian Butterflies dataset; good template to swap in your own small image dataset
- **Paper (optional, once comfortable):** "Scalable Diffusion Models with Transformers" (DiT paper) — read after Unit 4 notebook, not before

### Build

- Work through Unit 1's from-scratch notebook fully by hand — don't skip to the `diffusers` library version until you've implemented the noise/denoise loop yourself once
- Train a small UNet diffusion model on a narrow, small image dataset (start tiny on purpose — this is about seeing the process work, not building production Stable Diffusion)
- Optional stretch tied to COLOR: prototype a "generate venue mood art" feature using a hosted text-to-image API (not your from-scratch model — use a real API for anything user-facing) wired into the Flutter app

### Checkpoint

A trained small diffusion model that generates recognizable (if rough) images in your chosen domain, plus a one-paragraph explanation in your own words of what the forward and reverse processes are actually doing mathematically.

---

## WEEK 6 — Capstone

Pick one (both leverage everything from Weeks 2-4 specifically):

### Option A — "COLOR Concierge"

RAG (Week 2) + tool-calling agent (Week 3) built directly into the COLOR stack — Flutter frontend, Node backend, Mongo/Postgres — that recommends venues based on mood, live data, and user preferences. Real feature, real founder pitch, real resume line.

### Option B — "Trading Signal Analyst Agent"

Apply Weeks 3-4 to your Solana bot: an agent that reasons over your labeled trade-outcome dataset (the 1,342-outcome set you already built), explains _why_ a signal fired in plain language, and flags anomalies. Pairs directly with your Zerve/HackerEarth ML pipeline experience.

### Capstone process

1. **Day 1-2:** Scope it down to something shippable in 5 days — pick ONE core user flow, not the whole vision
2. **Day 3-4:** Build, using whichever of Weeks 1-5's techniques actually serve the use case (you likely won't need all of them — that's fine, real engineering is knowing what to leave out)
3. **Day 5:** Write it up like a portfolio piece — problem, architecture diagram, tradeoffs you made, what you'd do differently with more time, and real numbers (latency, cost per query, eval scores) — this is what turns a side project into an interview talking point

### Checkpoint

A shipped, working feature with a written case-study-style summary you could hand to a Handshake Fellowship reviewer or a hiring manager without further editing.

---

## Weekly rhythm (suggested)

| Day     | Focus                                                                                                     |
| ------- | --------------------------------------------------------------------------------------------------------- |
| Mon-Tue | Read/watch the week's core resource, take notes in your own words                                         |
| Wed-Thu | Build                                                                                                     |
| Fri     | Build + start eval/checkpoint                                                                             |
| Weekend | Finish checkpoint, write a short summary of what you learned (this is what you'll draw on for interviews) |

Adjust around your Handshake Fellowship and Fedstack OA prep load — if a week needs to stretch to 10 days because Project Vox or Project Ivy work takes priority, that's a completely reasonable trade.
