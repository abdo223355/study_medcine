"""
test_retrieval.py -- measure the BASELINE of vector retrieval (no LLM, no tuning).

    python test_retrieval.py

Prints the top-5 chunks for 6 questions and a summary table. "Expected pages" are where the answer
lives in Pharmacology.pdf (found by reading the PDF), so we can check the hits against them.
"""
import rag

# (question, expected pages in Pharmacology.pdf)
QUERIES = [
    ("What is pharmacology?",                                       [9]),
    ("What is the difference between pharmacology and pharmacy?",   [10]),
    ("Explain second messenger cascades.",                          [41]),
    ("What are the types of receptors?",                            [53, 54, 55, 56, 57]),
    ("ما هو علم الأدوية؟",                                           [9]),
    ("ما الفرق بين علم الأدوية والصيدلة؟",                            [10]),
]


def show(question, results, expected):
    print(f"\n{'=' * 78}\nQ: {question}   (expected pages: {expected})\n{'=' * 78}")
    for rank, r in enumerate(results, start=1):
        hit = "  <== expected page" if r["page"] in expected else ""
        preview = r["text"].replace("\n", " / ")
        preview = preview if len(preview) <= 230 else preview[:230] + "..."
        print(f"{rank}. {r['chunk_id']:22s} page={r['page']:<4d} distance={r['distance']:.4f}{hit}")
        print(f"   {preview}")


def main():
    rows = []
    for question, expected in QUERIES:
        results = rag.search_chunks(question, top_k=5, where={"subject": "Pharmacology"})
        show(question, results, expected)
        pages = [r["page"] for r in results]
        first = next((i for i, p in enumerate(pages, start=1) if p in expected), None)
        rows.append((question, expected, pages, first))

    print(f"\n\n{'Query':58s} | {'Expected':10s} | {'Top-5 pages':22s} | First hit rank")
    print("-" * 110)
    for question, expected, pages, first in rows:
        exp = f"{expected[0]}-{expected[-1]}" if len(expected) > 1 else str(expected[0])
        print(f"{question[:58]:58s} | {exp:10s} | {str(pages):22s} | {first if first else 'MISS'}")


if __name__ == "__main__":
    main()
