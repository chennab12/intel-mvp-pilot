"""Start Here: orientation and the 15-question diagnostic."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from modules.ui import ACCENT, content, goto, page_header, progress
from utils.calculators import diagnostic_by_topic, recommend_path, score_answers

PATH_LABELS = {"beginner": "Beginner", "working": "Working TPM", "senior": "Senior TPM"}


def render() -> None:
    c = content()
    page_header(
        "🚀 Start Here",
        "This academy helps a new TPM in AI/ML Customer Engineering understand the technical foundations, "
        "follow engineering discussions, run customer evaluations, and build senior-level judgment. "
        "Everything runs on an ordinary laptop; no Intel accelerator, cloud account or API key needed.",
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Lessons", len(c["lessons"]))
    col2.metric("Practical labs", len(c["labs"]))
    col3.metric("Knowledge checks", sum(len(lesson["checks"]) for lesson in c["lessons"]))

    with st.expander("How to use this app (2 minutes)", expanded=False):
        st.markdown(
            "1. **Take the diagnostic below.** It recommends a learning path.\n"
            "2. **Follow a Learning Path.** Every lesson has a plain-English explanation, a customer example, a visual, "
            "an exercise, misconceptions, senior-TPM questions, a knowledge check and sources.\n"
            "3. **Practice.** Use Practical Labs, the Customer POC Simulator, the Performance Playground and the "
            "Troubleshooting Clinic.\n"
            "4. **Use the Senior TPM Toolkit** templates in real work (replace the fictional examples).\n"
            "5. **Save progress** from *My Progress* as a JSON file. Progress lives only in this browser session.\n\n"
            "**Labels you'll see:** 💻 runs on a normal laptop · 🧪 simulation only · 🔌 requires specified Intel hardware. "
            "Data is always labeled *measured*, *computed*, *simulated*, *vendor claim* or *needs verification*."
        )
    with st.expander("About accuracy, sources and confidentiality"):
        st.markdown(
            "- Product facts come only from public sources listed in **Glossary and Sources**, each with the date it was "
            "actually checked. Facts that could not be checked are marked **Needs verification**.\n"
            "- All customers, people and numbers in examples are **fictional**.\n"
            "- This is an independent learning tool. It is not official Intel training and contains no confidential "
            "Intel or customer information. Don't paste confidential material into it.\n"
            "- Completing lessons builds skills; it does not guarantee a promotion."
        )

    st.subheader("Recommended first three lessons")
    st.caption("Picked for a learner with ops/SRE/monitoring experience and limited coding background.")
    cols = st.columns(3)
    for col, lid in zip(cols, c["recommended_first"]):
        lesson = c["lessons_by_id"][lid]
        with col.container(border=True):
            st.markdown(f"**{lesson['title']}**")
            st.caption(lesson["summary"])
            st.button("Open lesson", key=f"rec_{lid}", on_click=goto, args=("Learning Paths",),
                      kwargs={"lesson_request": lid, "lp_view": "Lessons"})

    st.divider()
    render_diagnostic(c)


def render_diagnostic(c: dict) -> None:
    diag = c["diagnostic"]
    questions = diag["questions"]
    st.subheader(f"📋 Diagnostic assessment ({len(questions)} questions)")
    st.caption("About 10 minutes. Answers are hidden until you submit. Your result recommends a starting path.")
    with st.form("diagnostic"):
        answers = []
        for i, q in enumerate(questions):
            answers.append(st.radio(f"**{i + 1}. {q['question']}**", options=list(range(len(q["options"]))),
                                    format_func=lambda j, q=q: q["options"][j], index=None, key=f"diag_{i}"))
        submitted = st.form_submit_button("Submit diagnostic", type="primary")
    if submitted:
        scored = score_answers(questions, answers)
        by_topic = diagnostic_by_topic(questions, answers)
        path = recommend_path(by_topic)
        progress()["diagnostic"] = {"by_topic": by_topic, "recommended_path": path,
                                    "correct": scored["correct"], "total": scored["total"]}
        st.session_state.diag_answers = answers
    result = progress().get("diagnostic")
    if not result:
        return
    st.success(f"Score: **{result['correct']} / {result['total']}**. Recommended path: "
               f"**{PATH_LABELS[result['recommended_path']]}**")
    topics = diag["topics"]
    names = [topics[t] for t in topics]
    pcts = [round(100 * result["by_topic"].get(t, {}).get("correct", 0) /
                  max(1, result["by_topic"].get(t, {}).get("total", 1))) for t in topics]
    fig = go.Figure(go.Bar(x=pcts, y=names, orientation="h", marker_color=ACCENT,
                           text=[f"{p}%" for p in pcts], textposition="outside"))
    fig.update_layout(height=280, margin=dict(l=10, r=30, t=10, b=10), xaxis=dict(range=[0, 110], title="% correct"))
    st.plotly_chart(fig, width="stretch")
    st.button(f"Start the {PATH_LABELS[result['recommended_path']]} path", type="primary",
              on_click=goto, args=("Learning Paths",),
              kwargs={"lp_path": result["recommended_path"], "lp_view": "Lessons"})

    answers = st.session_state.get("diag_answers")
    if answers:
        with st.expander("Review answers and explanations"):
            for i, (q, a) in enumerate(zip(questions, answers)):
                ok = a == q["answer"]
                st.markdown(f"{'✅' if ok else '❌'} **{i + 1}. {q['question']}**")
                st.markdown(f"Answer: **{q['options'][q['answer']]}**. {q['explanation']}")
                lesson = c["lessons_by_id"][q["lesson"]]
                st.caption(f"Topic: {topics[q['topic']]} · Lesson: {lesson['title']}")
