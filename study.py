"""
study.py -- study features on top of the retrieval in rag.py: summary, MCQ generation, quiz grading.

    summarize(topic)             -> {"summary", "found", "sources"}
    generate_mcqs(topic, n=5)    -> {"questions": [...], "found", "sources"}
    grade_quiz(questions, answers) -> score + per-question feedback   (pure code, no LLM)

Same safety rule as answer_question: nothing relevant retrieved -> found=False and the LLM is not called.
"""
import json
import random
import re

import rag

LETTERS = "ABCD"

SUMMARY_PROMPT = (
    "You are a study assistant for medical students. Write a clear, structured summary of the TOPIC using ONLY "
    "the CONTEXT. Use short headings and bullet points. Do not add outside knowledge. "
    "If the context says nothing about the topic, reply exactly: NOT_FOUND. "
    "Write in the same language as the topic (Arabic or English); keep medical terms in English. "
    "Never say \"the context\"; say \"the lecture notes\" if you must refer to the source. "
    "Cover every part of the context, in the order it appears. "
    "If something is cut off or missing (for example a formula), write: not covered in the lecture notes. "
    "Never write \"notes extract\". Skip table rows that carry no information. "
    "After each key point add its source like [Pharmacology.pdf p.9]."
)

MCQ_PROMPT = (
    "You write exam questions for medical students. Create up to {n} multiple-choice questions using ONLY the "
    "CONTEXT. Do not use outside knowledge. Each question tests one fact stated in the context, and the questions must cover DIFFERENT parts of the context. "
    "Never write \"according to the context\" in a question or explanation. "
    "Return ONLY a JSON array (no markdown, no text around it). Each item: "
    '{{"question": str, "options": [4 different strings, the CORRECT answer FIRST, 3 plausible wrong answers], '
    '"explanation": one or two sentences taken from the context, "page": the page number from the label of the '
    "chunk you used (an integer)}}. "
    "If the context cannot support any question, return []. "
    "Write in the same language as the topic (Arabic or English); keep medical terms in English."
)


def _summarize_chunks(chunks, topic, chat):
    chat = chat or rag._chat
    if not chunks:
        return rag.not_found_result(topic, summary=None)
    text = chat(SUMMARY_PROMPT, f"CONTEXT:\n{rag.build_context(chunks)}\n\nTOPIC: {topic}")
    if "NOT_FOUND" in text:
        return rag.not_found_result(topic, summary=None)
    return {"summary": text, "found": True, "sources": rag.cited_sources(text, chunks)}


def summarize(topic, where=None, chat=None):
    """Summary of a free-text topic: uses the top-5 most relevant chunks (search)."""
    return _summarize_chunks(rag.relevant_chunks(topic, where=where), topic, chat)


def summarize_lesson(subject, lesson, chat=None):
    """Summary of a whole lesson: uses ALL its chunks (no search), so nothing from the lesson is left out."""
    return _summarize_chunks(rag.get_lesson_chunks(subject, lesson), lesson, chat)


def _parse_json_list(text):
    """The model should return a bare JSON array, but tolerate ```json fences or a sentence around it."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def _build_question(item, source_by_page, rng):
    """Validate one item from the LLM and turn it into a quiz question. Returns None if anything is wrong."""
    if not isinstance(item, dict):
        return None
    question, options = item.get("question"), item.get("options")
    explanation, page = item.get("explanation"), item.get("page")
    if not (isinstance(question, str) and question.strip() and isinstance(explanation, str)):
        return None
    if not (isinstance(options, list) and len(options) == 4 and all(isinstance(o, str) and o.strip() for o in options)):
        return None
    if len({o.strip().lower() for o in options}) != 4:
        return None
    if isinstance(page, str) and page.isdigit():
        page = int(page)
    if page not in source_by_page:                 # the page must be one we really retrieved -> no invented sources
        return None
    correct = options[0]
    shuffled = options[:]
    rng.shuffle(shuffled)                          # the LLM puts the right answer first; the code mixes the order
    return {
        "question": question.strip(),
        "options": dict(zip(LETTERS, shuffled)),
        "answer": LETTERS[shuffled.index(correct)],
        "explanation": explanation.strip(),
        "source": source_by_page[page],
        "page": page,
    }


def _mcqs_from_chunks(chunks, topic, n, chat, rng):
    chat = chat or rag._chat
    rng = rng or random.Random()
    if not chunks:
        return {**rag.not_found_result(topic), "questions": []}
    source_by_page = {c["page"]: c["source"] for c in chunks}
    user = f"CONTEXT:\n{rag.build_context(chunks)}\n\nTOPIC: {topic}\nNumber of questions: {n}"

    items = None
    for _ in range(2):                             # one retry if the reply is not valid JSON
        items = _parse_json_list(chat(MCQ_PROMPT.format(n=n), user))
        if items is not None:
            break
    questions, seen = [], set()
    for item in items or []:
        q = _build_question(item, source_by_page, rng)
        if q and q["question"].lower() not in seen:
            seen.add(q["question"].lower())
            q["id"] = len(questions) + 1
            questions.append(q)
        if len(questions) == n:
            break
    if not questions:
        return {**rag.not_found_result(topic), "questions": []}
    sources = [{"source": s, "page": p} for p, s in sorted(source_by_page.items())
               if p in {q["page"] for q in questions}]
    return {"questions": questions, "found": True, "sources": sources}


def generate_mcqs(topic, n=5, where=None, chat=None, rng=None):
    """MCQs on a free-text topic: uses the top-5 most relevant chunks (search)."""
    return _mcqs_from_chunks(rag.relevant_chunks(topic, where=where), topic, n, chat, rng)


def generate_lesson_mcqs(subject, lesson, n=5, chat=None, rng=None):
    """MCQs from a whole lesson: uses ALL its chunks, so questions can come from any part of it."""
    return _mcqs_from_chunks(rag.get_lesson_chunks(subject, lesson), lesson, n, chat, rng)


def grade_quiz(questions, answers):
    """answers: {question id: "A".."D"}. Unanswered counts as wrong. Pure code, no LLM."""
    results, score = [], 0
    for q in questions:
        given = str(answers.get(q["id"], "")).strip().upper()
        ok = given == q["answer"]
        score += ok
        results.append({
            "id": q["id"], "question": q["question"], "your_answer": given or None,
            "correct_answer": q["answer"], "correct_text": q["options"][q["answer"]],
            "correct": ok, "explanation": q["explanation"], "source": q["source"], "page": q["page"]})
    total = len(questions)
    return {"score": score, "total": total, "percent": round(100 * score / total) if total else 0, "results": results}
