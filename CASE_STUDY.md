# Journal Intelligence Agent — Case Study

**Project:** AI-powered RAG system over 2.5 years of personal journal entries  
**Author:** Adam Seeman  
**Built with:** Python, Sentence Transformers, ChromaDB, Claude API  
**Timeline:** May 2026

---

## The Problem

I have 232 journal entries spanning October 2023 to April 2026 — covering professional cycling, quitting sport, traveling through 15+ countries, starting an AI degree, and hitchhiking South America. That's a lot of lived experience buried in a text file.

The problem: I couldn't search my own life meaningfully. Keyword search finds words, not meaning. I wanted to ask real questions: *"When did I feel most lost and what pulled me out?"* or *"What patterns show up before bad mental health weeks?"*

---

## The Solution: RAG (Retrieval Augmented Generation)

RAG solves a core limitation of LLMs: **they can't know things you never told them**, and you can't tell them everything at once (too expensive, context limits).

The pipeline:

```
Journal entries → chunked → embedded → stored in vector DB
                                                  ↓
User question → embedded → semantic search → top N relevant entries
                                                  ↓
                               Claude reads those entries → answers
```

### Key concepts implemented:

**1. Embeddings**  
Each journal entry is converted into a vector — a list of ~384 numbers that encodes its *meaning*. Entries about loneliness cluster near each other in vector space. Entries about racing cluster somewhere else. This is done by a model called `all-MiniLM-L6-v2`, running entirely locally (no API cost).

**2. Vector Search**  
When you ask a question, it's converted into a vector too. ChromaDB finds the stored entry-vectors that are mathematically closest to the question-vector. "Closest" means "most semantically similar."

**3. Generation**  
Claude receives your question + the retrieved entries as context, and generates an answer grounded in your actual journal data.

---

## Technical Stack

| Component | Tool | Why |
|---|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Free, runs locally, fast |
| Vector store | ChromaDB | Simple, persistent, local |
| LLM | Claude (Anthropic API) | Best reasoning quality |
| Language | Python 3.11 | Standard for AI/ML |

---

## Data

- **Source:** DayOne journal export (single .txt file)
- **Entries:** 232 entries, Oct 2023 – Apr 2026
- **Locations covered:** Czech Republic, France, Spain, Philippines, Vietnam, Nepal, India, Central Asia, China, Singapore, Australia, South America
- **Domains:** Cycling career, mental health, relationships, travel, learning, identity

---

## Results & Example Queries

### Query: "What did professional sport teach me that could be my unfair advantage in a working environment?"

Both backends received identical context: **4,622 characters across 5 retrieved journal entries**.

---

**Gemma 3 4B (local, free)**

> *"The most compelling evidence comes from his breakdown during the ride (Entry 3). He explicitly states 'just simply because of the anxiety and stress'. However, he also notes he 'tried to remember what happened last year'. This shows a capacity for processing setbacks..."*

Gemma identified 4 themes (resilience, focus, pressure management, small wins) and structured them clearly. It stayed close to the surface of the entries and asked a follow-up question at the end.

**Verdict:** Competent. Good factual retrieval. Stays generic.

---

**Claude Opus (API)**

> *"Don't sell it as 'I was a pro athlete so I'm disciplined.' Sell it as: I've already had my career-ending crisis, I know how I break, and I know how I rebuild. That's a 35-year-old's wisdom in a 22-year-old."*

Claude cited specific entries by date, identified a non-obvious pattern (identity = performance as the *biggest* advantage), and ended with actionable framing for interviews. It also flagged an honest caveat from the entries.

**Verdict:** Noticeably sharper. Better pattern recognition, more nuanced synthesis.

---

### Backend Comparison

| | Gemma 3 4B (local) | Claude Opus (API) |
|---|---|---|
| Cost | Free | ~$0.01–0.05 per query |
| Privacy | 100% local — data never leaves machine | Sent to Anthropic's servers |
| Speed | ~15–20s on 8GB RAM | ~3–5s |
| Quality | Good for factual Q&A | Better for pattern recognition and nuanced synthesis |
| Best for | Development, private use, cost-sensitive deployment | Production, when answer quality matters |

**Key insight:** The retrieval step (embeddings + ChromaDB) is identical for both. The quality difference is entirely in the generation step. This architecture makes swapping backends trivial — one environment variable.

---

## What I Learned

- **RAG is a retrieval problem as much as a generation problem.** Garbage retrieval gives garbage context, and even the best LLM can't recover from that.
- **Chunking strategy matters.** Choosing entry-level chunks (vs fixed-size) preserves emotional and temporal context that would otherwise be split mid-thought.
- **Local models are viable for development** — Gemma 3 4B on 8GB RAM handles this task well enough to build and test without API costs.
- **The architecture separates concerns cleanly:** ChromaDB never talks to an LLM; the LLM never touches the database. Your Python code is the only bridge, which makes the system easy to reason about and modify.
- **Privacy is a real design decision.** Running embeddings locally with sentence-transformers and generation locally with Ollama means the journal data never leaves the machine — a meaningful choice for personal data.

---

## Code

GitHub: *(link to be added)*
