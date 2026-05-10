# Journal Intelligence Agent

A RAG (Retrieval Augmented Generation) system that lets you query 2.5 years of personal journal entries using natural language. Ask questions about patterns, emotions, decisions, and experiences — grounded in actual journal data.

## What it does

Instead of keyword search, this uses semantic search: entries about loneliness are found even if you use completely different words than what was written. The system finds the most relevant entries, then sends them to an LLM to synthesize a grounded answer.

**Example queries:**
- *"When did I feel most lost and what pulled me out of it?"*
- *"What patterns show up before bad mental health weeks?"*
- *"What was I thinking about during my time in Nepal?"*
- *"What did professional sport teach me that translates to a work environment?"*

## Architecture

```
Journal entries (raw text)
        ↓
   [ingest.py] Parse → Embed → Store in ChromaDB
        ↓
   ChromaDB (local vector database)
        ↓
User question → Embed → Semantic search → Top 5 entries
        ↓
   LLM (Claude or Gemma) → Grounded answer
```

**Key design decisions:**
- **Entry-level chunking** — each journal entry is one chunk, preserving emotional and temporal context
- **Local embeddings** — `all-MiniLM-L6-v2` runs entirely on device, no API cost for search
- **Swappable LLM backend** — Claude (quality) or Gemma 3 4B via Ollama (free, private, local)
- **Conversation memory** — context is maintained across turns within a session

## Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector store | ChromaDB (local, persistent) |
| LLM (cloud) | Claude Opus via Anthropic API |
| LLM (local) | Gemma 3 4B via Ollama |
| Language | Python 3.11 |

## Setup

**1. Install dependencies**
```bash
pip install sentence-transformers chromadb anthropic
```

**2. For local model (optional)**
Install [Ollama](https://ollama.com) then:
```bash
ollama pull gemma3:4b
```

**3. Set environment variables**
```bash
export ANTHROPIC_API_KEY=your_key   # only needed for Claude backend
```

**4. Ingest your journal data**

Export your journal as a plain text file and update the `JOURNAL_FILE` path in `ingest.py`, then:
```bash
python ingest.py
```

This runs once and builds the local vector database (~30 seconds for 200 entries).

**5. Start chatting**
```bash
python main.py                   # Claude (default)
python main.py --backend ollama  # Gemma 3 4B (local, free)
python main.py --verbose         # show retrieval debug info
```

## Commands

| Input | Action |
|---|---|
| Any question | Query your journal |
| `sources` | Show which entries were retrieved for the last question |
| `clear` | Reset conversation memory |
| `quit` | Exit |

## Backend comparison

|  | Gemma 3 4B (local) | Claude Opus (API) |
|---|---|---|
| Cost | Free | ~$0.01–0.05 per query |
| Privacy | 100% local | Sent to Anthropic servers |
| Speed | ~15–20s on 8GB RAM | ~3–5s |
| Quality | Good for factual Q&A | Better for nuanced synthesis |

See [CASE_STUDY.md](CASE_STUDY.md) for a full comparison with example outputs.

## Project context

Built as part of a portfolio demonstrating applied AI / RAG systems. The data is 232 personal journal entries spanning October 2023 – April 2026, covering professional cycling, travel across 15+ countries, and an AI degree.
