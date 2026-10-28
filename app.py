"""
app.py -- the dashboard.   Run:  streamlit run app.py

Sidebar: choose subject and lesson.   Tabs: Summary | Quiz | Ask.
All the logic lives in rag.py and study.py; this file only draws the screens.
"""
import os
import sys

# Streamlit Community Cloud ships an old sqlite3 that ChromaDB refuses to use; swap in a newer one if installed.
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st

import rag
import study

THIN_LESSON_CHARS = 1000        # a lesson with less text than this gives weak summaries / few questions

st.set_page_config(page_title="Medical Study Assistant", page_icon="🩺", layout="wide")
# Arabic answers read right-to-left, English ones left-to-right (direction is decided per paragraph).
st.markdown("<style>.stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 "
            "{unicode-bidi: plaintext; text-align: start;}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- optional password (for a hosted copy)
# If APP_PASSWORD is set (hosting "Secrets" / .env), nothing is shown or sent to the LLM until it is typed.
# Not set (local use) -> no password. Keeps strangers from spending the OpenRouter credit.
_password = os.environ.get("APP_PASSWORD")
if _password and not st.session_state.get("authed"):
    st.title("🩺 Medical Study Assistant")
    typed = st.text_input("Password", type="password")
    if typed:
        if typed == _password:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()

@st.cache_data(show_spinner=False)
def subjects():
    return rag.list_subjects()


@st.cache_data(show_spinner=False)
def lessons(subject):
    return rag.list_lessons(subject)


def sources_line(sources):
    return "Sources: " + ", ".join(f"{s['source']} p.{s['page']}" for s in sources) if sources else ""


def not_found_box():
    st.warning("Nothing about this was found in the uploaded files.")


# ---------------------------------------------------------------- sidebar
st.title("🩺 Medical Study Assistant")
try:
    all_subjects = subjects()
except Exception as e:                      # no database yet
    st.error(f"Could not open the vector database: {e}")
    st.stop()
if not all_subjects:
    st.error("The database is empty. Run:  python ingest.py data/<file>.pdf --build")
    st.stop()

subject = st.sidebar.selectbox("Subject", all_subjects)
lesson_list = lessons(subject)
by_name = {l["lesson"]: l for l in lesson_list}
lesson = st.sidebar.selectbox("Lesson", list(by_name))
info = by_name[lesson]
st.sidebar.caption(f"{info['chunks']} text chunks · {info['chars']} characters")
if info["chars"] < THIN_LESSON_CHARS:
    st.sidebar.warning("This lesson has very little text (most of it is probably images), "
                       "so the results will be weak.")
st.sidebar.caption("Lessons that are only images (no text) are not listed.")
key = (subject, lesson)

tab_summary, tab_quiz, tab_ask = st.tabs(["📝 Summary", "❓ Quiz", "💬 Ask"])

# ---------------------------------------------------------------- summary
with tab_summary:
    st.subheader(lesson)
    if st.button("Generate summary", type="primary"):
        with st.spinner("Summarizing the lesson..."):
            try:
                st.session_state["summary"] = (key, study.summarize_lesson(subject, lesson))
            except Exception as e:
                st.error(f"Error: {e}")
    saved = st.session_state.get("summary")
    if saved and saved[0] == key:
        result = saved[1]
        if result["found"]:
            st.markdown(result["summary"])
            st.caption(sources_line(result["sources"]))
        else:
            not_found_box()

# ---------------------------------------------------------------- quiz
with tab_quiz:
    st.subheader(lesson)
    n = st.number_input("Number of questions", min_value=1, max_value=10, value=5, step=1)
    if st.button("Generate quiz", type="primary"):
        with st.spinner("Writing questions..."):
            try:
                result = study.generate_lesson_mcqs(subject, lesson, n=int(n))
                st.session_state["quiz"] = {"key": key, "result": result, "graded": None, "version": 0}
            except Exception as e:
                st.error(f"Error: {e}")

    quiz = st.session_state.get("quiz")
    if quiz and quiz["key"] == key:
        result = quiz["result"]
        if not result["found"]:
            not_found_box()
        else:
            questions = result["questions"]
            if len(questions) < n:
                st.info(f"Only {len(questions)} valid questions could be written from this lesson.")
            with st.form(f"quiz-form-{quiz['version']}"):
                for q in questions:
                    st.markdown(f"**{q['id']}. {q['question']}**")
                    st.radio("Your answer", list(q["options"]), index=None, key=f"ans-{quiz['version']}-{q['id']}",
                             format_func=lambda letter, q=q: f"{letter}) {q['options'][letter]}",
                             label_visibility="collapsed")
                submitted = st.form_submit_button("Submit answers")
            if submitted:
                answers = {q["id"]: st.session_state.get(f"ans-{quiz['version']}-{q['id']}") or "" for q in questions}
                quiz["graded"] = study.grade_quiz(questions, answers)

            graded = quiz["graded"]
            if graded:
                st.success(f"Score: {graded['score']} / {graded['total']}  ({graded['percent']}%)")
                for x in graded["results"]:
                    icon = "✅" if x["correct"] else "❌"
                    with st.expander(f"{icon} Question {x['id']}: {x['question']}", expanded=not x["correct"]):
                        st.markdown(f"Your answer: **{x['your_answer'] or 'not answered'}** · "
                                    f"Correct answer: **{x['correct_answer']}) {x['correct_text']}**")
                        st.markdown(x["explanation"])
                        st.caption(f"{x['source']} p.{x['page']}")
                if st.button("New quiz on this lesson"):
                    quiz["version"] += 1
                    st.session_state["quiz"] = None
                    st.rerun()

# ---------------------------------------------------------------- ask
ASK_ENABLED = os.environ.get("ENABLE_ASK", "true").lower() != "false"   # Summary and Quiz never need the embedding model
with tab_ask:
    if not ASK_ENABLED:
        st.info("The Ask tab needs the large embedding model, so it is switched off on this free demo host. "
                "Summary and Quiz are fully available.")
        st.stop()
    st.caption(f"Ask anything about {subject}. Answers come only from the uploaded files.")
    with st.form("ask-form"):
        question = st.text_input("Your question (Arabic or English)")
        asked = st.form_submit_button("Ask", type="primary")
    if asked and question.strip():
        with st.spinner("Searching the notes..."):
            try:
                st.session_state["answer"] = (question, rag.answer_question(question, where={"subject": subject}))
            except Exception as e:
                st.error(f"Error: {e}")
    saved = st.session_state.get("answer")
    if saved:
        _, result = saved
        if result["found"]:
            st.markdown(result["answer"])
            st.caption(sources_line(result["sources"]))
        else:
            st.warning(result["answer"])
