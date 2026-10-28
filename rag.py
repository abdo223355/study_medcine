"""
rag.py -- embeddings + ChromaDB (persistent, local).
Retrieval only: no LLM, no prompt, no answer generation yet.

    build_vector_store(chunks)  : embed chunks once and save them to db/chroma
    search_chunks(query, ...)   : embed the query and return the top_k closest chunks
"""
import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")    # reads OPENROUTER_API_KEY / OPENROUTER_MODEL from the .env file

# ---------------------------------------------------------------- settings
# To change the embedding model: change this one line, then rebuild the database
# (vectors from different models are not comparable; get_collection() refuses to mix them).
EMBEDDING_MODEL = "BAAI/bge-m3"

DB_PATH = Path(__file__).parent / "db" / "chroma"
COLLECTION_NAME = "medical_chunks"      # ONE collection for all subjects; subject is a metadata field


# ---------------------------------------------------------------- embeddings (the only model-specific code)
_model = None


def embed_texts(texts, batch_size=16):
    """list[str] -> list[list[float]]. Vectors are L2-normalised, so cosine distance = 1 - dot product.

    bge-m3 needs no 'query:' / 'passage:' prefix. (A model that does, e.g. multilingual-e5, would need one added here.)
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer   # imported lazily: loading takes seconds
        _model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = _model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                            show_progress_bar=len(texts) > 32)
    return vectors.tolist()


# ---------------------------------------------------------------- ChromaDB
def get_collection():
    """Open (or create) the persistent collection. Cosine space; remembers which model made the vectors."""
    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return client.create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL})

    stored = (collection.metadata or {}).get("embedding_model")
    if stored != EMBEDDING_MODEL:
        raise RuntimeError(
            f"{DB_PATH} was built with '{stored}' but rag.py is set to '{EMBEDDING_MODEL}'. "
            f"Rebuild the database (delete db/chroma, then run: python ingest.py <pdf> --build).")
    return collection


def _metadata(c):
    return {"source": c["source"], "page": c["page"], "chunk_id": c["chunk_id"],
            "subject": c["subject"], "kind": c["kind"], "lesson": c["lesson"]}


def build_vector_store(chunks, rebuild=False):
    """
    Embed `chunks` (output of chunk_pages) and store them. Chunks already in the database are skipped,
    so running it again costs nothing. rebuild=True first deletes the stored chunks of these sources.
    """
    collection = get_collection()
    if rebuild:
        for source in {c["source"] for c in chunks}:
            collection.delete(where={"source": source})

    existing = set(collection.get(ids=[c["chunk_id"] for c in chunks])["ids"])
    new = [c for c in chunks if c["chunk_id"] not in existing]
    old = [c for c in chunks if c["chunk_id"] in existing]
    if old:   # refresh metadata of chunks already stored (cheap, no re-embedding) -- e.g. a newly added field
        collection.update(ids=[c["chunk_id"] for c in old], metadatas=[_metadata(c) for c in old])
    if new:
        collection.add(
            ids=[c["chunk_id"] for c in new],                       # chunk_id = Chroma document id
            documents=[c["text"] for c in new],
            embeddings=embed_texts([c["text"] for c in new]),
            metadatas=[_metadata(c) for c in new],
        )
    return {"added": len(new), "already_in_db": len(existing), "total_in_db": collection.count()}


def search_chunks(query, top_k=5, where=None, include_exam=False):
    """
    query -> embedding -> Chroma similarity search -> top_k chunks (closest first).
    where : optional Chroma metadata filter, e.g. {"subject": "Pharmacology"}.
    include_exam : by default exam/exercise chunks are left out (they outrank the lecture text); pass True to search them.
    distance is cosine distance: 0 = identical direction, smaller = more similar.
    """
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError("The vector store is empty. Run: python ingest.py <pdf> --build")
    if not include_exam:
        kind = {"kind": "content"}
        where = kind if not where else {"$and": [where, kind]}

    result = collection.query(
        query_embeddings=embed_texts([query]),
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"chunk_id": cid, "text": text, "page": meta["page"], "source": meta["source"],
         "subject": meta["subject"], "kind": meta.get("kind"), "distance": round(dist, 4)}
        for cid, text, meta, dist in zip(result["ids"][0], result["documents"][0],
                                         result["metadatas"][0], result["distances"][0])
    ]


# ---------------------------------------------------------------- lessons (a lesson = one chapter of the PDF)
def list_subjects():
    """Subjects that are in the database, e.g. ["Microbiology", "Pharmacology"]."""
    return sorted({m["subject"] for m in get_collection().get(include=["metadatas"])["metadatas"]})


def list_lessons(subject):
    """Lessons of a subject that have study text, in chapter order: [{"lesson", "chunks", "chars"}]."""
    got = get_collection().get(where={"$and": [{"subject": subject}, {"kind": "content"}]},
                               include=["documents", "metadatas"])
    lessons = {}
    for doc, meta in zip(got["documents"], got["metadatas"]):
        entry = lessons.setdefault(meta["lesson"], {"lesson": meta["lesson"], "chunks": 0, "chars": 0})
        entry["chunks"] += 1
        entry["chars"] += len(doc)
    def order(e):   # "Course overview" first, then Chapter 1, 2, ...
        m = re.match(r"Chapter (\d+)", e["lesson"])
        return int(m.group(1)) if m else 0
    return sorted(lessons.values(), key=order)


def get_lesson_chunks(subject, lesson):
    """ALL content chunks of one lesson, in page order (no similarity search: a lesson is small, we use all of it)."""
    got = get_collection().get(where={"$and": [{"subject": subject}, {"kind": "content"}, {"lesson": lesson}]},
                               include=["documents", "metadatas"])
    chunks = [{"chunk_id": m["chunk_id"], "text": d, "page": m["page"], "source": m["source"],
               "subject": m["subject"], "kind": m["kind"], "lesson": m["lesson"], "distance": 0.0}
              for d, m in zip(got["documents"], got["metadatas"])]
    return sorted(chunks, key=lambda c: (c["page"], c["chunk_id"]))


# ---------------------------------------------------------------- LLM answer (OpenRouter)
# One API key, OpenAI-compatible. Put it in the .env file next to this script (OPENROUTER_API_KEY=...), never in code.
TOP_K = 5
LLM_MODEL = os.environ.get("OPENROUTER_MODEL") or "openai/gpt-4o-mini"   # change the model here or via env var
MAX_DISTANCE = 0.52        # chunks farther than this are treated as "not related". From the baseline: good hits
                           # were <= 0.49, unrelated ones >= 0.56. Re-tune with out-of-scope questions.
NOT_FOUND_AR = "لم أجد الإجابة في المحتوى المرفوع."
NOT_FOUND_EN = "I could not find the answer in the uploaded content."

SYSTEM_PROMPT = (
    "You are a study assistant for medical students. Answer ONLY from the CONTEXT below. "
    "Do not use outside knowledge and do not guess. "
    "If the context does not contain the answer, reply exactly: NOT_FOUND. "
    "Answer in the same language as the question (Arabic or English); keep medical terms in English. "
    "Refer to the source as \"the lecture notes\", never as \"the context\". "
    "After each fact add its source like [Pharmacology.pdf p.9]. Be clear and concise."
)


def build_context(chunks):
    """Chunks -> one text block; every chunk is labelled so the LLM can cite file + page."""
    return "\n\n".join(f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in chunks)


def _chat(system, user):
    """One LLM call through OpenRouter. Returns the reply text."""
    from openai import OpenAI                      # OpenRouter speaks the OpenAI protocol
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is empty. Put your key in the .env file.")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    resp = client.chat.completions.create(
        model=LLM_MODEL, temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return resp.choices[0].message.content.strip()


def is_arabic(text):
    return any("\u0600" <= ch <= "\u06ff" for ch in text)


def not_found_result(question, **extra):
    return {"answer": NOT_FOUND_AR if is_arabic(question) else NOT_FOUND_EN, "found": False, "sources": [], **extra}


def relevant_chunks(query, top_k=TOP_K, where=None):
    """Retrieve, then drop weak chunks (farther than MAX_DISTANCE). Empty list = nothing relevant in the files."""
    return [c for c in search_chunks(query, top_k=top_k, where=where) if c["distance"] <= MAX_DISTANCE]


def cited_sources(text, chunks):
    """Sources the text actually cites ("[file p.N]"), one entry per page, best match first. Fallback: all chunks."""
    cited = {(m.group(1), int(m.group(2))) for m in re.finditer(r"\[([^\]]+?)\s+p\.(\d+)\]", text)}
    used = [c for c in chunks if (c["source"], c["page"]) in cited] or chunks
    sources, seen = [], set()
    for c in used:
        if (c["source"], c["page"]) not in seen:
            seen.add((c["source"], c["page"]))
            sources.append({"source": c["source"], "page": c["page"], "distance": c["distance"]})
    return sources


def answer_question(question, top_k=TOP_K, where=None, chat=None):
    """question -> retrieve -> drop weak chunks -> context -> LLM -> {"answer", "found", "sources"}."""
    chat = chat or _chat
    chunks = relevant_chunks(question, top_k, where)
    if not chunks:
        return not_found_result(question)          # LLM is not even called
    text = chat(SYSTEM_PROMPT, f"CONTEXT:\n{build_context(chunks)}\n\nQUESTION: {question}")
    if "NOT_FOUND" in text:
        return not_found_result(question)
    return {"answer": text, "found": True, "sources": cited_sources(text, chunks)}
