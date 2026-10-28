"""
test_study.py -- end-to-end test of lessons, summary, MCQ generation and quiz grading (needs OPENROUTER_API_KEY).

First (once):   python ingest.py data/Pharmacology.pdf --build      (adds the lesson field, no re-embedding)
                python ingest.py data/Microbiology.pdf --build      (embeds the second subject)
Then:           python test_study.py
"""
import rag
import study


def show_questions(result):
    for q in result["questions"]:
        print(f"\n{q['id']}. {q['question']}")
        for letter, text in q["options"].items():
            print(f"   {letter}) {text}")
        print(f"   [answer: {q['answer']} | {q['source']} p.{q['page']}] {q['explanation']}")


def main():
    print("######## LESSONS PER SUBJECT ########")
    for subject in ("Pharmacology", "Microbiology"):
        print(subject)
        for l in rag.list_lessons(subject):
            print(f"   {l['lesson']}   ({l['chunks']} chunks, {l['chars']} chars)")

    ch4 = "Chapter 4: Principles of Drug-Body Interactions"
    print(f"\n######## LESSON SUMMARY: {ch4} ########")
    r = study.summarize_lesson("Pharmacology", ch4)
    print(f"found={r['found']}\n{r['summary']}\nsources: {[s['page'] for s in r['sources']]}")

    ch5 = "Chapter 5: Classification of Drug Receptors"
    print(f"\n######## LESSON MCQs (English): {ch5}, n=5 ########")
    r = study.generate_lesson_mcqs("Pharmacology", ch5, n=5)
    print(f"found={r['found']}  questions={len(r['questions'])}  pages={[s['page'] for s in r['sources']]}")
    show_questions(r)
    questions = r["questions"]

    print("\n######## LESSON MCQs, second subject: Microbiology Chapter 2: Fungi, n=3 ########")
    r = study.generate_lesson_mcqs("Microbiology", "Chapter 2: Fungi", n=3)
    print(f"found={r['found']}  questions={len(r['questions'])}")
    show_questions(r)

    print("\n######## NOT AVAILABLE (expect found=False, no LLM call) ########")
    ch6 = "Chapter 3: Drug Sources and Routes of Administration"   # image-only slides, no text
    print("Chapter 3 summary:", study.summarize_lesson("Pharmacology", ch6)["found"])
    print("metformin MCQs   :", study.generate_mcqs("Mechanism of action of metformin",
                                                   where={"subject": "Pharmacology"})["found"])

    print("\n######## QUIZ GRADING (first 3 correct, the rest wrong, last one left empty) ########")
    answers = {q["id"]: q["answer"] for q in questions[:3]}
    for q in questions[3:-1]:
        answers[q["id"]] = next(l for l in "ABCD" if l != q["answer"])
    g = study.grade_quiz(questions, answers)
    print(f"score {g['score']}/{g['total']} ({g['percent']}%)")
    for x in g["results"]:
        print(f"  Q{x['id']}: {'OK ' if x['correct'] else 'WRONG'} your={x['your_answer']} correct={x['correct_answer']} ({x['source']} p.{x['page']})")


if __name__ == "__main__":
    main()
