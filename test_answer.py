"""
test_answer.py -- end-to-end test: question -> retrieval -> LLM -> answer + sources.
4 questions that ARE in the PDF, 2 that are NOT (the system must say "not found").

    python test_answer.py        (needs OPENROUTER_API_KEY)
"""
import rag

IN_PDF = [
    "What is pharmacology?",
    "What is the difference between pharmacology and pharmacy?",
    "What are the types of receptors?",
    "ما الفرق بين علم الأدوية والصيدلة؟",
]
NOT_IN_PDF = [
    "What is the capital of France?",                       # far away: the distance filter should stop it
    "How do I treat diabetic ketoacidosis in the ICU?",
]
# Medical, close to the lecture topics, but NOT in the PDF: chunks pass the filter, so the LLM itself must say NOT_FOUND.
NEAR_MISS = [
    "What is the mechanism of action of metformin?",
    "ما هي آلية عمل الميتفورمين؟",
    "What is the pediatric dose of amoxicillin for otitis media?",   # amoxicillin is only mentioned in passing
]


def main():
    for label, questions in (("IN PDF", IN_PDF), ("NOT IN PDF (expect found=False)", NOT_IN_PDF),
                             ("NEAR MISS: medical but not in PDF (expect found=False)", NEAR_MISS)):
        print(f"\n######## {label} ########")
        for q in questions:
            r = rag.answer_question(q, where={"subject": "Pharmacology"})
            print(f"\nQ: {q}\nfound={r['found']}\n{r['answer']}")
            print("sources:", [(s["source"], s["page"], s["distance"]) for s in r["sources"]])


if __name__ == "__main__":
    main()
