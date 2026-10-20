"""Intel AI/ML Technical TPM Academy - Streamlit entrypoint.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Intel AI/ML Technical TPM Academy", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

from modules import explorers, labs_page, learning_paths, performance, simulators, start_here, toolkit  # noqa: E402
from modules import agent_lab_page, model_testing  # noqa: E402
from modules import cheatsheet  # noqa: E402
from modules import concepts_math  # noqa: E402
from modules.ui import content, goto, progress  # noqa: E402
from utils.search import search  # noqa: E402

SECTIONS = {
    "Start Here": ("🚀", start_here.render),
    "Learning Paths": ("🗺️", learning_paths.render),
    "Intel Product Explorer": ("🧩", explorers.render_products),
    "Software Stack Explorer": ("🧱", explorers.render_stack),
    "Practical Labs": ("🧪", labs_page.render),
    "Customer POC Simulator": ("🧭", simulators.render_poc),
    "Performance Playground": ("📈", performance.render),
    "Troubleshooting Clinic": ("🩺", simulators.render_clinic),
    "Senior TPM Toolkit": ("🧰", toolkit.render_toolkit),
    "My Progress": ("📊", toolkit.render_progress),
    "Glossary and Sources": ("📚", toolkit.render_glossary),
    "Model Functional Testing": ("✅", model_testing.render_functional),
    "Model Performance Testing": ("⏱️", model_testing.render_performance),
    "Agentic AI Inference Lab": ("🤖", agent_lab_page.render),
    "Golden Master Cheatsheet": ("🧾", cheatsheet.render),
    "Senior TPM Concepts & Inference Math": ("🎓", concepts_math.render),
}

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-size: 16px; }
      .block-container { max-width: 1100px; padding-top: 2rem; }
      h1 { font-size: 2.0rem !important; } h2 { font-size: 1.5rem !important; } h3 { font-size: 1.2rem !important; }
      p, li { line-height: 1.6; }
    </style>
    """,
    unsafe_allow_html=True,
)


def sidebar() -> str:
    c = content()
    prog = progress()
    with st.sidebar:
        st.markdown("## 🎓 Intel AI/ML TPM Academy")
        st.caption("Independent learning app · public information only · fictional examples")
        st.session_state.setdefault("nav", "Start Here")
        section = st.radio("Go to", list(SECTIONS), key="nav",
                           format_func=lambda s: f"{SECTIONS[s][0]} {s}", label_visibility="collapsed")
        done = len(prog["completed_lessons"])
        total = len(c["lessons"])
        st.progress(done / total, text=f"{done} of {total} lessons complete")
        st.text_input("🔎 Search everything", key="global_search", placeholder="KV cache, Gaudi, NUMA...")
        st.caption("Progress lives in this session. Export it from My Progress.")
    return section


def search_results(query: str) -> None:
    results = search(query)
    with st.container(border=True):
        st.markdown(f"**Search results for “{query}”** ({len(results)})")
        if not results:
            st.caption("No matches. Try a shorter or different word.")
        for i, r in enumerate(results):
            a, b = st.columns([5, 1])
            a.markdown(f"**{r['title']}** · *{r['kind']}*  \n{r['summary'][:180]}")
            kwargs = {"global_search": ""}
            if r["kind"] == "Lesson":
                kwargs |= {"lesson_request": r["id"], "lp_view": "Lessons"}
            elif r["kind"] == "Lab":
                kwargs |= {"lab_pick": r["id"], "lab_labels": []}
            elif r["kind"] == "Troubleshooting case":
                kwargs |= {"case_pick": r["id"]}
            elif r["kind"] == "Scenario":
                kwargs |= {"poc_pick": r["id"]}
            elif r["kind"] == "Stack":
                kwargs |= {"stack_pick": r["id"]}
            elif r["kind"] == "Glossary":
                kwargs |= {"gl_view": "Glossary", "gl_q": r["id"]}
            elif r["kind"] == "Source":
                kwargs |= {"gl_view": "Source registry"}
            elif r["kind"] == "Product":
                kwargs |= {"pe_text": r["title"].split(",")[0][:40], "pe_cat": [], "pe_status": []}
            b.button("Open", key=f"sr_{i}_{r['id']}", on_click=goto, args=(r["section"],), kwargs=kwargs)


def main() -> None:
    section = sidebar()
    query = st.session_state.get("global_search", "").strip()
    if query:
        search_results(query)
    SECTIONS[section][1]()


main()
