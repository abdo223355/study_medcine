"""
compare_embeddings.py -- which embedding model retrieves best on OUR questions?

    python compare_embeddings.py

For each candidate model: embed the content chunks into a temporary Chroma collection (nothing touches db/chroma),
run the 6 test questions with top_k=5, and score them. Models are downloaded the first time (a few GB in total).
Score = how many expected pages appear in the top-5 (recall) + rank of the first expected page.
"""
import chromadb
from sentence_transformers import SentenceTransformer

import ingest
from test_retrieval import QUERIES

# (model name, prefix for questions, prefix for chunks) -- e5 models REQUIRE these prefixes, bge-m3 does not.
CANDIDATES = [
    ("BAAI/bge-m3",                                              "",       ""),
    ("intfloat/multilingual-e5-large",                           "query: ", "passage: "),
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "",     ""),
]
TOP_K = 5


def evaluate(name, q_prefix, p_prefix, chunks):
    model = SentenceTransformer(name)
    col = chromadb.EphemeralClient().create_collection("t", metadata={"hnsw:space": "cosine"})
    texts = [p_prefix + c["text"] for c in chunks]
    col.add(ids=[c["chunk_id"] for c in chunks], documents=[c["text"] for c in chunks],
            embeddings=model.encode(texts, batch_size=16, normalize_embeddings=True).tolist(),
            metadatas=[{"page": c["page"]} for c in chunks])
    rows = []
    for question, expected in QUERIES:
        res = col.query(query_embeddings=model.encode([q_prefix + question], normalize_embeddings=True).tolist(),
                        n_results=TOP_K, include=["metadatas"])
        pages = [m["page"] for m in res["metadatas"][0]]
        found = len({p for p in pages if p in expected})
        first = next((i for i, p in enumerate(pages, 1) if p in expected), None)
        rows.append((question, pages, found, len(expected), first))
    return rows


def main():
    chunks = [c for c in ingest.chunk_pages(ingest.extract_pages("data/Pharmacology.pdf"), "Pharmacology.pdf")
              if c["kind"] == "content"]
    print(f"{len(chunks)} content chunks\n")
    summary = []
    for name, qp, pp in CANDIDATES:
        print(f"=== {name}")
        try:
            rows = evaluate(name, qp, pp, chunks)
        except Exception as e:
            print(f"   FAILED: {e}\n")
            continue
        for q, pages, found, total, first in rows:
            print(f"   {q[:45]:45s} top5={pages}  expected found {found}/{total}  first hit rank={first}")
        recall = sum(r[2] / r[3] for r in rows) / len(rows)
        mrr = sum((1 / r[4]) if r[4] else 0 for r in rows) / len(rows)
        summary.append((name, recall, mrr))
        print()
    print("SUMMARY (higher is better)")
    print(f"{'model':65s} recall@5   MRR")
    for name, recall, mrr in sorted(summary, key=lambda s: (-s[1], -s[2])):
        print(f"{name:65s} {recall:.2f}      {mrr:.2f}")


if __name__ == "__main__":
    main()
