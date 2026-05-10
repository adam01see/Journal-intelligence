"""
The RAG query pipeline:
1. Embed the user's question (same model used during ingest)
2. Search ChromaDB for the most semantically similar journal entries
3. Send those entries + conversation history to Claude or Gemma via Ollama
4. Return the answer grounded in actual journal data
"""
import os
import requests
import chromadb
import anthropic
from sentence_transformers import SentenceTransformer

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "journal"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

BACKEND = os.environ.get("JOURNAL_BACKEND", "claude")
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

_model = None
_collection = None

SYSTEM_PROMPT = """You are an AI assistant with access to Adam's personal journal entries.
Adam is a 22-year-old Czech guy — former pro cyclist, now traveling nomad and AI student.

When answering:
- Ground every claim in the actual journal entries provided
- Quote or reference specific entries when relevant (mention the date)
- Be honest if the entries don't fully answer the question
- Be direct and insightful — Adam wants real reflection, not generic advice
- Look for patterns across multiple entries when they exist
- You have access to the conversation history — use it to give contextually aware answers"""


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Find the journal entries most semantically similar to the question."""
    model = _get_model()
    collection = _get_collection()

    question_vector = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    entries = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        entries.append({
            "text": doc,
            "date": meta.get("date", "Unknown"),
            "location": meta.get("location", "Unknown"),
            "similarity": round(1 - dist, 3),
        })
    return entries


def _build_context(retrieved: list[dict]) -> str:
    context = ""
    for i, entry in enumerate(retrieved, 1):
        context += f"\n--- Entry {i} | {entry['date']} | {entry['location']} ---\n"
        context += entry["text"] + "\n"
    return context


def _answer_claude(question: str, context: str, history: list[dict]) -> str:
    client = anthropic.Anthropic()

    # Build messages: history + new question with fresh context
    messages = list(history) + [
        {
            "role": "user",
            "content": f"Relevant journal entries for this question:\n{context}\n\nQuestion: {question}",
        }
    ]

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


def _answer_ollama(question: str, context: str, history: list[dict]) -> str:
    # Format conversation history into the prompt
    history_text = ""
    for msg in history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{history_text}"
        f"Relevant journal entries:\n{context}\n\n"
        f"Human: {question}\nAssistant:"
    )

    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def answer(question: str, retrieved: list[dict], history: list[dict], verbose: bool = False) -> str:
    context = _build_context(retrieved)

    if verbose:
        print(f"\n[DEBUG] Backend: {BACKEND} | {len(context)} chars | {TOP_K} entries | {len(history)//2} previous turns\n")

    if BACKEND == "ollama":
        return _answer_ollama(question, context, history)
    return _answer_claude(question, context, history)


def ask(question: str, history: list[dict] = None, verbose: bool = False) -> tuple[str, list[dict]]:
    """
    Full RAG pipeline: retrieve → answer.
    Returns (reply, updated_history) so the caller can maintain conversation state.
    """
    if history is None:
        history = []

    retrieved = retrieve(question)
    reply = answer(question, retrieved, history, verbose=verbose)

    # Append this turn to history for next call
    updated_history = list(history) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": reply},
    ]

    return reply, updated_history, retrieved
