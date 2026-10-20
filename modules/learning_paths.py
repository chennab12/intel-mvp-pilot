"""Learning Paths: browse lessons by path or track, read a lesson, take its knowledge check."""
from __future__ import annotations

import random

import streamlit as st

from modules.ui import content, goto, label_badge, page_header, progress, render_sources, render_visual
from utils.calculators import score_answers
from utils.content import EXERCISE_LABELS, all_checks
from utils.progress import completion_pct

ALL = "All lessons (by track)"


def render() -> None:
    c = content()
    page_header("🗺️ Learning Paths", "Pick a path for your level, or browse all lessons by track. "
                "Filters and the sidebar search help you find a topic quickly.")
    view = st.radio("View", ["Lessons", "Practice quiz"], horizontal=True, key="lp_view", label_visibility="collapsed")
    if view == "Practice quiz":
        render_practice_quiz(c)
        return

    path_options = [p["id"] for p in c["learning_paths"]] + [ALL]
    path_names = {p["id"]: p["name"] for p in c["learning_paths"]} | {ALL: ALL}
    st.session_state.setdefault("lp_path", "beginner")
    wanted = st.session_state.pop("lesson_request", None)  # set by links elsewhere in the app
    if wanted in c["lessons_by_id"]:
        st.session_state.lesson_pick = wanted
        for key, empty in (("lp_tracks", []), ("lp_labels", []), ("lp_status", "Any"), ("lp_text", "")):
            st.session_state[key] = empty
        current = next((p for p in c["learning_paths"] if p["id"] == st.session_state.lp_path), None)
        if current and wanted not in current["lessons"]:
            st.session_state.lp_path = ALL
    path_id = st.selectbox("Learning path", path_options, format_func=path_names.get, key="lp_path")

    if path_id == ALL:
        lesson_ids = [lesson["id"] for lesson in c["lessons"]]
    else:
        path = next(p for p in c["learning_paths"] if p["id"] == path_id)
        lesson_ids = path["lessons"]
        st.caption(f"{path['who']} Suggested pace: about {path['weeks']} weeks at 3-4 hours per week.")

    prog = progress()
    pct = completion_pct(prog, lesson_ids)
    st.progress(pct / 100, text=f"{pct}% of this path complete")

    with st.expander("Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        tracks = f1.multiselect("Track", list(c["tracks"]), format_func=lambda t: c["tracks"][t]["name"], key="lp_tracks")
        labels = f2.multiselect("Exercise type", EXERCISE_LABELS, key="lp_labels")
        status = f3.selectbox("Status", ["Any", "Not completed", "Completed"], key="lp_status")
        text = st.text_input("Filter by keyword", key="lp_text", placeholder="for example: KV cache, NUMA, Gaudi")

    lessons = [c["lessons_by_id"][lid] for lid in lesson_ids]
    if tracks:
        lessons = [lesson for lesson in lessons if lesson["path"] in tracks]
    if labels:
        lessons = [lesson for lesson in lessons if lesson["exercise"]["label"] in labels]
    if status == "Completed":
        lessons = [lesson for lesson in lessons if lesson["id"] in prog["completed_lessons"]]
    elif status == "Not completed":
        lessons = [lesson for lesson in lessons if lesson["id"] not in prog["completed_lessons"]]
    if text:
        t = text.lower()
        lessons = [lesson for lesson in lessons if t in (lesson["title"] + lesson["summary"] + lesson["explanation"]).lower()]

    if not lessons:
        st.info("No lessons match these filters. Clear a filter to see more.")
        return

    ids = [lesson["id"] for lesson in lessons]
    if st.session_state.get("lesson_pick") not in ids:
        st.session_state.lesson_pick = ids[0]

    def fmt(lid):
        lesson = c["lessons_by_id"][lid]
        mark = "✅" if lid in prog["completed_lessons"] else "◻️"
        return f"{mark} {c['tracks'][lesson['path']]['icon']} {lesson['title']}"

    lid = st.selectbox(f"Lesson ({len(ids)} shown)", ids, format_func=fmt, key="lesson_pick")
    st.divider()
    render_lesson(c, c["lessons_by_id"][lid], ids)


def render_lesson(c: dict, lesson: dict, ids: list[str]) -> None:
    prog = progress()
    lid = lesson["id"]
    st.header(lesson["title"])
    st.caption(f"Track {c['tracks'][lesson['path']]['name']} · Level: {lesson['level']} · About {lesson['minutes']} min · "
               + ("Product-specific (see sources)" if lesson["product_specific"] else "Concept lesson"))
    st.markdown(f"*{lesson['summary']}*")

    st.subheader("1. Plain-English explanation")
    st.markdown(lesson["explanation"])
    st.subheader("2. Why it matters to an Intel Customer Engineering TPM")
    st.markdown(lesson["why_it_matters"])
    st.subheader("3. Customer example (fictional)")
    st.info(lesson["customer_example"])
    st.subheader("4. Visual")
    render_visual(lesson["visual"])

    st.subheader("5. Practical exercise")
    ex = lesson["exercise"]
    label_badge(ex["label"], ex.get("hardware"))
    st.markdown(f"**{ex['title']}**")
    for i, step in enumerate(ex.get("steps", []), 1):
        st.markdown(f"{i}. {step}")
    if ex.get("code"):
        st.caption("Commands and code are shown for learning. This app never runs commands you type.")
        st.code(ex["code"], language="bash" if "lscpu" in ex["code"] or "pip3" in ex["code"] or "docker" in ex["code"] else "python")
    if ex.get("lab"):
        lab = next(lab for lab in c["labs"] if lab["id"] == ex["lab"])
        target = lab.get("link") or "Practical Labs"
        st.button(f"Open: {lab['title']}", key=f"lab_{lid}", on_click=goto, args=(target,),
                  kwargs={"lab_pick": lab["id"], "lab_labels": []} if target == "Practical Labs" else {})

    st.subheader("6. Expected observations or illustrative output")
    st.code(lesson["expected_output"].strip(), language="text", wrap_lines=True)
    st.subheader("7. How to interpret the result")
    st.markdown(lesson["interpretation"])
    st.subheader("8. Common misconceptions")
    for m in lesson["misconceptions"]:
        st.markdown(f"- {m}")
    st.subheader("9. Questions a senior TPM should ask")
    for q in lesson["senior_questions"]:
        st.markdown(f"- {q}")

    st.subheader("10. Knowledge check")
    render_checks(lesson)

    st.subheader("11. Sources and verification")
    render_sources(lesson["sources"], heading="Sources used in this lesson")

    st.divider()
    done = lid in prog["completed_lessons"]
    b1, b2, b3 = st.columns(3)
    if done:
        if b1.button("Mark as not complete", key=f"undone_{lid}"):
            prog["completed_lessons"].remove(lid)
            st.rerun()
    elif b1.button("✅ Mark lesson complete", key=f"done_{lid}", type="primary"):
        prog["completed_lessons"].append(lid)
        st.rerun()
    pos = ids.index(lid)
    if pos > 0:
        b2.button("← Previous lesson", key=f"prev_{lid}", on_click=goto, args=("Learning Paths",),
                  kwargs={"lesson_pick": ids[pos - 1]})
    if pos < len(ids) - 1:
        b3.button("Next lesson →", key=f"next_{lid}", on_click=goto, args=("Learning Paths",),
                  kwargs={"lesson_pick": ids[pos + 1]})


def render_checks(lesson: dict) -> None:
    lid = lesson["id"]
    checks = lesson["checks"]
    answers = []
    for i, q in enumerate(checks):
        answers.append(st.radio(f"**Q{i + 1}. {q['question']}**", list(range(len(q["options"]))),
                                format_func=lambda j, q=q: q["options"][j], index=None, key=f"kc_{lid}_{i}"))
    if st.button("Check my answers", key=f"kc_btn_{lid}"):
        st.session_state[f"kc_checked_{lid}"] = True
        scored = score_answers(checks, answers)
        progress()["check_results"][lid] = {"correct": scored["correct"], "total": scored["total"]}
    if st.session_state.get(f"kc_checked_{lid}"):
        res = progress()["check_results"].get(lid)
        if res:
            st.markdown(f"**Score: {res['correct']} / {res['total']}**")
        for i, (q, a) in enumerate(zip(checks, answers)):
            if a is None:
                st.warning(f"Q{i + 1}: not answered.")
            elif a == q["answer"]:
                st.success(f"Q{i + 1}: correct. {q['explanation']}")
            else:
                st.error(f"Q{i + 1}: not quite. {q['explanation']}")
    for i, q in enumerate(checks):
        with st.expander(f"Show answer to Q{i + 1}"):
            st.markdown(f"**{q['options'][q['answer']]}**. {q['explanation']}")


def render_practice_quiz(c: dict) -> None:
    st.subheader("📝 Practice quiz")
    st.caption("Ten questions drawn from lesson knowledge checks. Answers are hidden until you submit.")
    tracks = st.multiselect("Tracks", list(c["tracks"]), default=list(c["tracks"]),
                            format_func=lambda t: c["tracks"][t]["name"], key="pq_tracks")
    pool = [q for q in all_checks(c) if q["path"] in tracks]
    if not pool:
        st.info("Choose at least one track.")
        return
    st.session_state.setdefault("pq_seed", 0)
    if st.button("New set of questions"):
        st.session_state.pq_seed += 1
        st.session_state.pop("pq_result", None)
    rng = random.Random(st.session_state.pq_seed)
    questions = rng.sample(pool, min(10, len(pool)))
    with st.form("practice_quiz"):
        answers = [st.radio(f"**{i + 1}. {q['question']}**", list(range(len(q["options"]))),
                            format_func=lambda j, q=q: q["options"][j], index=None,
                            key=f"pq_{st.session_state.pq_seed}_{i}") for i, q in enumerate(questions)]
        if st.form_submit_button("Submit quiz", type="primary"):
            st.session_state.pq_result = (score_answers(questions, answers), answers)
    if "pq_result" in st.session_state:
        scored, answers = st.session_state.pq_result
        st.success(f"Score: {scored['correct']} / {scored['total']} ({scored['pct']}%)")
        for i, (q, a, ok) in enumerate(zip(questions, answers, scored["results"])):
            st.markdown(f"{'✅' if ok else '❌'} **{i + 1}.** {q['question']}  \n"
                        f"Answer: **{q['options'][q['answer']]}**. {q['explanation']}  \n"
                        f"*From lesson: {q['lesson_title']}*")
