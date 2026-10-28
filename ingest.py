"""
ingest.py -- PDF -> pages (extract_pages) -> chunks (chunk_pages).
(No embeddings, ChromaDB or LLM yet.)
"""
import math
import re
import sys
from pathlib import Path

import pymupdf  # PyMuPDF (the older import name is `fitz`; both are the same library)

INVISIBLE_CHARS = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
WORD_GAP_RATIO = 0.4   # gap between glyphs bigger than 0.4 x font size = a real space between words
FAKE_GAP_MAX_RATIO = 0.22   # letter-spacing gaps are ~0.1-0.19 x font size
GAP_TOLERANCE = 0.3         # points: the gaps inside one spaced run are identical
REAL_SPACE_MARGIN = 0.12    # a real word space must be this x font size wider than the fake gaps


def clean_text(text):
    """Very light cleaning: no rewording, only invisible chars and whitespace noise."""
    text = text.translate(INVISIBLE_CHARS).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)                 # repeated spaces/tabs -> one space
    text = re.sub(r" *\n *", "\n", text)                # spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)              # many blank lines -> one
    return text.strip()


def is_letter_spaced(text):
    """'O r i g i n s a n d T e r m s' -> most tokens on the line are single characters."""
    tokens = text.split()
    return len(tokens) >= 5 and sum(len(t) == 1 for t in tokens) / len(tokens) >= 0.7


def drop_fake_spaces(glyphs):
    """
    For lines that are NOT letter-spaced as a whole but contain a spaced run, e.g. 'C o r e' or '(E m a x)'.

    A candidate run = >= 2 consecutive space-separated glyph pairs whose gaps are tiny
    (< 0.22 x font size) and equal (+-0.3pt). That alone also matches normal justified text with a
    one-letter word ("describes a drug's"), so a run is accepted only with extra evidence:
      (a) every glyph on the line is separated by a space (the whole line is spaced), or
      (b) the same line has a REAL word space clearly wider than the run's gaps
          (fake gaps ~0.1-0.19 x size, real spaces ~0.3-0.4 x size).
    Only the fake space characters are removed; nothing else changes.
    """
    pairs, prev, spaces = [], None, []          # pairs: (space char indices between 2 glyphs, gap, size, punct?)
    for i, (c, size) in enumerate(glyphs):
        if c["c"].isspace():
            spaces.append(i)
            continue
        if prev is not None:
            joined_by_punct = not (prev[0]["c"].isalnum() and c["c"].isalnum())   # e.g. "(E", "x)", ")."
            pairs.append((spaces, c["bbox"][0] - prev[0]["bbox"][2], prev[1], joined_by_punct))
        prev, spaces = (c, size), []

    def is_candidate(sp, gap, size):
        return bool(sp) and 0 <= gap <= FAKE_GAP_MAX_RATIO * size


    runs, run = [], []
    for sp, gap, size, _ in pairs + [([], 0, 1, False)]:          # sentinel closes the last run
        if is_candidate(sp, gap, size) and (not run or abs(gap - run[0][1]) <= GAP_TOLERANCE):
            run.append((sp, gap, size))
            continue
        if len(run) >= 2:
            runs.append(run)
        run = [(sp, gap, size)] if is_candidate(sp, gap, size) else []

    whole_line_spaced = bool(pairs) and all(sp or punct for sp, _, _, punct in pairs)
    drop = set()
    for run in runs:
        run_gap, size = run[0][1], run[0][2]
        in_run = {id(r[0]) for r in run}
        has_wider_real_space = any(
            sp and id(sp) not in in_run and gap >= run_gap + REAL_SPACE_MARGIN * size
            for sp, gap, _, _ in pairs
        )
        if whole_line_spaced or has_wider_real_space:
            drop.update(i for sp, _, _ in run for i in sp)
    return "".join(c["c"] for i, (c, _) in enumerate(glyphs) if i not in drop)


def rebuild_line(line):
    """
    Some PDFs put a space after every glyph, so the real word boundaries are lost in the text.
    The glyph positions still know them: inside a word the gap is small, between words it is large.
    Returns (text, was_modified).
    """
    glyphs = [(c, span["size"]) for span in line["spans"] for c in span["chars"]]
    text = "".join(c["c"] for c, _ in glyphs)

    if not is_letter_spaced(text):
        fixed = drop_fake_spaces(glyphs)         # partial spacing inside an otherwise normal line
        return fixed, fixed != text

    # Whole line is letter-spaced: drop the fake spaces and re-insert one only at a large gap.
    size = max(span["size"] for span in line["spans"])
    out, prev = "", None
    for c, _ in glyphs:
        if c["c"].isspace():
            continue
        if prev is not None and c["bbox"][0] - prev["bbox"][2] > WORD_GAP_RATIO * size:
            out += " "
        out += c["c"]
        prev = c
    return out, True


def extract_pages(pdf_path):
    """Return [{"page": 1, "text": "..."}, ...]. `page` is the real 1-based PDF page number."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            lines = []
            for block in page.get_text("rawdict")["blocks"]:
                if block["type"] != 0:          # 0 = text block, 1 = image block
                    continue
                for line in block["lines"]:
                    lines.append(rebuild_line(line)[0])
            pages.append({"page": page.number + 1, "text": clean_text("\n".join(lines))})
    return pages


# ====================================================================== chunking
# Chunking works on the TEXT of each page only (output of extract_pages); it never touches the PDF.
MAX_CHARS = 800         # ~200 tokens. See the explanation in the chat; tune after retrieval tests.
MIN_PAGE_CHARS = 100    # pages with less text than this (blank / title-only dividers) produce no chunks
# Pages that are explicitly NOT study material (Table of Contents). Per file, chosen by hand.
SKIP_PAGES = {"Pharmacology.pdf": {2, 3, 4, 5}}

BULLETS = "\u25cf\u2022\u25e6\u25aa\u25a0\u25cb\u00b7"                 # ● • ◦ ▪ ■ ○ ·
SENTENCE_END = re.compile(r"[.!?:;][\"')\]]*$")                         # line ends a sentence/item
LIST_MARK = re.compile(r"^(\d+\s*[.)]|[A-Za-z]\))\s")                    # "1. ", "12) ", "a) "
LABEL = re.compile(r"^[A-Z][^:]{0,40}:\s")                               # "Etymology: ...", "Core Concept: ..."
SENTENCE_BREAK = re.compile(r"[.!?][\"')\]]*\s+")


def group_lines(text):
    """
    Group the lines of a page into units: one bullet, one list item, one labelled definition,
    or one paragraph. PDF lines are wrapped visually, so a line that does not end a sentence
    continues on the next line. The lines themselves are kept unchanged.
    """
    units, prev = [], None
    for line in text.split("\n"):
        if not line.strip():
            prev = None                                # blank line = paragraph break
            continue
        starts_new = (
            prev is None
            or line[0] in BULLETS
            or LIST_MARK.match(line)
            or LABEL.match(line)
            or SENTENCE_END.search(prev)
        )
        if starts_new:
            units.append(line)
        else:
            units[-1] += "\n" + line
        prev = line
    return units


def split_long_unit(text, budget):
    """Fallback for ONE unit longer than the budget: cut at sentence ends, repeat the last sentence as overlap."""
    if len(text) <= budget:
        return [text]
    cuts = [0] + [m.end() for m in SENTENCE_BREAK.finditer(text)] + [len(text)]
    sentences = [text[a:b] for a, b in zip(cuts, cuts[1:]) if b > a]
    parts, cur = [], []
    for sentence in sentences:
        if cur and len("".join(cur) + sentence) > budget:
            parts.append("".join(cur).strip())
            cur = cur[-1:]                             # overlap = the last sentence only
        cur.append(sentence)
    parts.append("".join(cur).strip())
    return parts


def pack_units(units, max_chars):
    """Pack units into groups of at most max_chars, balanced (not 800 + 30)."""
    pieces = [p for u in units for p in split_long_unit(u, max_chars)]
    total = len("\n".join(pieces))
    target = total / max(1, math.ceil(total / max_chars))

    groups, cur = [], []
    for piece in pieces:
        cur_len = len("\n".join(cur))
        if cur and (cur_len >= target or cur_len + 1 + len(piece) > max_chars):
            groups.append(cur)
            cur = []
        cur.append(piece)
    if cur:
        groups.append(cur)
    return groups


# Exam / exercise sections (MCQ, true-false, short notes, matching). They repeat the lecture's terms, so in
# retrieval they outrank the real explanation. They are tagged kind="exam" (kept: useful later for MCQ ideas).
# A section starts on a page whose first line looks like "I - Choose...", "II. True / False", "1- Define the
# following"; it ends on the first page that starts with a lecture heading such as "4.1 ...".
EXAM_START = re.compile(r"^(?:(?:I|II|III|IV|V|VI)\s*[-.]\s*(?:Choose|Complete|True|Give|Match)|1\s*-\s*Define the following)", re.I)
LECTURE_HEADING = re.compile(r"^\d+\.\d+\b")


def parse_toc(pages):
    """
    Chapters listed in the table of contents (first pages of the file): [{"num": 4, "title": "Principles of ..."}].
    Works for PDFs whose TOC has lines like "Chapter 4: Principles of Drug-Body Interactions". None found -> [].
    """
    toc_text = " ".join(" ".join(p["text"].split()) for p in pages[:8])
    chapters, seen = [], set()
    for m in re.finditer(r"Chapter\s*(\d+)\s*:\s*(.*?)(?=\s*◦|\s*•|\s*SECTION|\s*Chapter\s*\d+\s*:|$)", toc_text):
        num = int(m.group(1))
        if num not in seen:
            seen.add(num)
            chapters.append({"num": num, "title": m.group(2).strip()})
    return chapters


def parse_dividers(pages):
    """
    Fallback for PDFs without a "Chapter N:" table of contents but with divider pages like
    "CHAPTER 2 / Fungi / ...": the chapter starts on that page. Returns [{"num", "title", "start"}].
    """
    chapters = []
    for p in pages:
        m = re.search(r"^\s*CHAPTER\s+(\d+)\s*\n\s*(.+(?:&|,|and)\s*\n.+|.+)", p["text"], re.M)   # may follow a stray icon line
        if m and len(p["text"]) < 300:                                      # divider pages are short
            chapters.append({"num": int(m.group(1)), "title": " ".join(m.group(2).split()), "start": p["page"]})
    return chapters


def assign_lessons(pages, chapters):
    """
    page number -> lesson name. A chapter starts on the first page that begins with its section heading
    ("4.1 ..."), searching only AFTER the table of contents. Pages before chapter 1 -> "Course overview".
    A chapter whose heading never appears as text (image-only slides) gets no pages.
    """
    toc_end = max((p["page"] for p in pages[:8] if re.search(r"\bChapter\s*\d+\s*:", p["text"])), default=0)
    starts = {}
    for ch in chapters:
        if "start" in ch:                       # divider pages already know where the chapter starts
            starts[ch["num"]] = ch["start"]
            continue
        for p in pages:
            if p["page"] > toc_end and re.match(rf"\s*{ch['num']}\.\d+\b", p["text"]):
                starts[ch["num"]] = p["page"]
                break
    order = sorted(starts.items(), key=lambda kv: kv[1])
    names = {c["num"]: f"Chapter {c['num']}: {c['title']}" for c in chapters}
    lesson_of = {}
    for p in pages:
        lesson = "Course overview"
        for num, start in order:
            if p["page"] >= start:
                lesson = names[num]
        lesson_of[p["page"]] = lesson
    return lesson_of


def chunk_pages(pages, source, max_chars=MAX_CHARS, skip_pages=None):
    """
    pages  : [{"page": 1, "text": "..."}, ...]   (output of extract_pages)
    source : file name, e.g. "Pharmacology.pdf"
    return : [{"chunk_id": "Pharmacology-p9-1", "text": "...", "page": 9,
               "source": "Pharmacology.pdf", "subject": "Pharmacology", "kind": "content" | "exam",
               "lesson": "Chapter 4: Principles of Drug-Body Interactions"}, ...]

    A chunk never crosses a page boundary, so `page` is always exact.
    skip_pages : pages to leave out (default: SKIP_PAGES[source]). They stay in extract_pages() output.
    """
    stem = Path(source).stem
    skip_pages = SKIP_PAGES.get(source, set()) if skip_pages is None else set(skip_pages)
    chunks = []
    in_exam = False
    lesson_of = assign_lessons(pages, parse_toc(pages) or parse_dividers(pages))
    for p in pages:
        text = p["text"].strip()
        if p["page"] in skip_pages or len(text) < MIN_PAGE_CHARS:
            continue
        if EXAM_START.match(text):
            in_exam = True
        elif LECTURE_HEADING.match(text):
            in_exam = False
        for n, group in enumerate(pack_units(group_lines(text), max_chars), start=1):
            chunks.append({
                "chunk_id": f"{stem}-p{p['page']}-{n}",
                "text": "\n".join(group),
                "page": p["page"],
                "source": source,
                "subject": stem,
                "kind": "exam" if in_exam else "content",
                "lesson": lesson_of[p["page"]],
            })
    return chunks


def find_image_only_pages(pdf_path):
    """
    For monitoring only: pages that have NO extractable text but contain images. Their content is not in the
    text corpus (no OCR, no descriptions). Also returns pages with images and very little text (< MIN_PAGE_CHARS).
    """
    image_only, little_text = [], []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            if not page.get_images():
                continue
            n_chars = len(page.get_text().strip())
            if n_chars == 0:
                image_only.append(page.number + 1)
            elif n_chars < MIN_PAGE_CHARS:
                little_text.append(page.number + 1)
    return image_only, little_text


if __name__ == "__main__":
    # python ingest.py data/Pharmacology.pdf            -> extract + chunk, print a summary
    # python ingest.py data/Pharmacology.pdf --build    -> ... then embed and save to db/chroma (once)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "data/Pharmacology.pdf"

    pages = extract_pages(path)
    chunks = chunk_pages(pages, Path(path).name)
    image_only, little_text = find_image_only_pages(path)
    lens = [len(c["text"]) for c in chunks]
    print(f"{path}: {len(pages)} pages -> {len(chunks)} chunks "
          f"(min {min(lens)}, avg {sum(lens) // len(lens)}, max {max(lens)} chars)")
    n_exam = sum(c["kind"] == "exam" for c in chunks)
    print(f"kind: {len(chunks) - n_exam} content, {n_exam} exam")
    toc = parse_toc(pages) or parse_dividers(pages)
    with_text = {c["lesson"] for c in chunks if c["kind"] == "content"}
    print(f"lessons found (table of contents or chapter divider pages): {len(toc)}")
    for ch in toc:
        name = f"Chapter {ch['num']}: {ch['title']}"
        n = sum(1 for c in chunks if c["lesson"] == name and c["kind"] == "content")
        print(f"   {name}  ->  {n} content chunks" + ("" if n else "   (NO TEXT: image-only slides, needs OCR)"))
    print(f"image_only_pages (no text, not in the corpus): {image_only}")
    print(f"image pages with very little text (skipped as short): {little_text}")
    assert not {c["page"] for c in chunks} & set(image_only), "a chunk came from an image-only page!"

    if "--build" in sys.argv:
        import rag
        print(f"embedding model: {rag.EMBEDDING_MODEL}")
        print(rag.build_vector_store(chunks, rebuild="--rebuild" in sys.argv))
        print(f"database: {rag.DB_PATH}")
