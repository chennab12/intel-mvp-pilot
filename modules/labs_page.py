"""Practical Labs: safe, predefined in-process exercises."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.ui import ACCENT, content, goto, label_badge, page_header, progress, render_result
from utils import labs as runners
from utils.content import EXERCISE_LABELS


def render() -> None:
    c = content()
    page_header("🧪 Practical Labs", "Hands-on exercises. 💻 laptop labs run here (or on your own laptop with the code "
                "shown); 🧪 simulations use toy models; 🔌 hardware labs list exactly what hardware you need. "
                "The app only runs predefined, safe Python functions. It never executes text you type or shell commands.")
    labels = st.multiselect("Type", EXERCISE_LABELS, key="lab_labels")
    labs = [lab for lab in c["labs"] if not labels or lab["label"] in labels]
    if not labs:
        st.info("No labs of that type.")
        return
    ids = [lab["id"] for lab in labs]
    if st.session_state.get("lab_pick") not in ids:
        st.session_state.lab_pick = ids[0]
    done = progress()["labs_done"]
    st.progress(len([i for i in done if i in ids]) / len(ids), text=f"{len([i for i in done if i in ids])} of {len(ids)} labs done")
    lab_id = st.selectbox("Lab", ids, key="lab_pick",
                          format_func=lambda i: ("✅ " if i in done else "") + next(lab["title"] for lab in labs if lab["id"] == i))
    lab = next(lab for lab in labs if lab["id"] == lab_id)

    st.header(lab["title"])
    label_badge(lab["label"], lab.get("hardware"))
    lesson = c["lessons_by_id"][lab["lesson"]]
    st.caption(f"About {lab['minutes']} min · Related lesson: {lesson['title']}")
    st.markdown(f"**Objective:** {lab['objective']}")
    for i, step in enumerate(lab["steps"], 1):
        st.markdown(f"{i}. {step}")
    if lab.get("code"):
        with st.expander("Code to run on your own laptop (optional)", expanded=lab["label"] != "Runs on a normal laptop"):
            st.code(lab["code"], language="bash" if lab["code"].lstrip().startswith(("pip", "hl-smi", "lscpu")) else "python")

    if lab.get("link"):
        st.button(f"Go to {lab['link']}", on_click=goto, args=(lab["link"],), type="primary")
    elif lab.get("runner"):
        st.markdown("#### Run it")
        params = lab_inputs(lab)
        if st.button("▶ Run", key=f"run_{lab_id}", type="primary"):
            try:
                st.session_state[f"lab_result_{lab_id}"] = runners.RUNNERS[lab["runner"]](params)
            except ValueError as exc:
                st.error(f"Input problem: {exc}")
        result = st.session_state.get(f"lab_result_{lab_id}")
        if result:
            render_result(result)
            lab_chart(lab, result)
            if lab["runner"] == "environment_inventory":
                st.download_button("Download inventory (JSON)", json.dumps(result["table"], indent=2),
                                   file_name="environment_inventory.json", mime="application/json")
    else:
        st.info("This lab needs the hardware listed above. Nothing is executed in this app; the expected output "
                "below is illustrative only.")

    st.markdown("#### Expected observations")
    st.markdown(lab["expected"])
    st.markdown("#### How to interpret")
    st.markdown(lab["interpretation"])
    if lab_id not in done:
        if st.button("Mark lab done", key=f"done_{lab_id}"):
            done.append(lab_id)
            st.rerun()
    st.button("Open related lesson", key=f"lesson_{lab_id}", on_click=goto, args=("Learning Paths",),
              kwargs={"lesson_request": lab["lesson"], "lp_view": "Lessons"})


def lab_inputs(lab: dict) -> dict:
    r = lab["runner"]
    k = lab["id"]
    if r == "bandwidth_vs_compute":
        return {"array_mb": st.slider("Array size (MB)", 10, 400, 100, 10, key=f"{k}_mb"),
                "matrix": st.select_slider("Matrix size", [256, 512, 1024, 2048], 1024, key=f"{k}_mat")}
    if r == "quantization_error":
        a, b = st.columns(2)
        return {"per_channel": a.toggle("Per-channel scaling", key=f"{k}_pc"),
                "outliers": b.toggle("Add 5 outlier channels", key=f"{k}_out")}
    if r == "token_counting":
        return {"text": st.text_area("Your text", "Our store assistant answers questions about coffee machines, "
                                     "recipes and opening hours. It must reply in under two seconds.", key=f"{k}_txt"),
                "chunks": st.slider("Retrieved chunks added", 0, 20, 4, key=f"{k}_ch"),
                "chunk_tokens": st.slider("Tokens per chunk (assumed)", 50, 2000, 300, 50, key=f"{k}_ct")}
    if r == "latency_percentiles":
        return {"runs": st.slider("Number of runs", 20, 1000, 200, 20, key=f"{k}_runs")}
    if r == "pipeline_bottleneck":
        cols = st.columns(3)
        return {"preprocess": cols[0].number_input("Preprocess work", 1000, 400000, 50000, 1000, key=f"{k}_a"),
                "compute": cols[1].number_input("Compute work", 1000, 400000, 200000, 1000, key=f"{k}_b"),
                "postprocess": cols[2].number_input("Postprocess work", 1000, 400000, 20000, 1000, key=f"{k}_c")}
    if r == "correctness_check":
        a, b = st.columns(2)
        return {"precision": a.selectbox("Candidate precision", ["FP16", "FP32"], key=f"{k}_p"),
                "tolerance": b.number_input("Tolerance (max abs difference)", 0.0, 1.0, 0.05, 0.001, format="%.3f", key=f"{k}_t")}
    if r == "shrink_mre":
        st.caption("The failing script. Untick lines to remove them.")
        kept = []
        for step, line in runners.MRE_STEPS:
            if st.checkbox(f"`{line}`", value=True, key=f"{k}_{step}"):
                kept.append(step)
        return {"kept": kept}
    if r == "critical_path":
        df = st.data_editor(pd.DataFrame(runners.DEFAULT_PLAN), num_rows="dynamic", key=f"{k}_plan", width="stretch",
                            column_config={"days": st.column_config.NumberColumn("days", min_value=0, max_value=365)})
        tasks = [row for row in df.to_dict("records") if str(row.get("task") or "").strip()]
        for row in tasks:
            row["days"] = float(row.get("days") or 0)
            row["depends_on"] = row.get("depends_on") or ""
        return {"tasks": tasks}
    if r == "requirement_check":
        return {"text": st.text_area("Requirement", "The system should be fast.", key=f"{k}_req")}
    if r == "update_checker":
        return {"text": st.text_area("Bottom line", "Things are going OK.", key=f"{k}_upd")}
    if r == "classify_workload":
        answers = []
        for i, (req, _, _) in enumerate(runners.CLASSIFY_ITEMS):
            answers.append(st.radio(req, runners.CLASSIFY_LABELS, index=None, horizontal=True, key=f"{k}_{i}"))
        return {"answers": answers}
    if r == "scaling_efficiency":
        a, b = st.columns(2)
        return {"comm": a.slider("Communication fraction of single-device step time", 0.0, 0.5, 0.1, 0.01, key=f"{k}_comm"),
                "devices": b.slider("Max devices", 2, 64, 16, key=f"{k}_dev")}
    return {}


def lab_chart(lab: dict, result: dict) -> None:
    if lab["runner"] == "latency_percentiles" and result.get("samples_ms"):
        fig = go.Figure(go.Histogram(x=result["samples_ms"], marker_color=ACCENT, nbinsx=40))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="ms per run (measured)",
                          yaxis_title="runs")
        st.plotly_chart(fig, width="stretch")
    if lab["runner"] == "scaling_efficiency" and result.get("table"):
        df = pd.DataFrame(result["table"])
        fig = go.Figure([go.Scatter(x=df["devices"], y=df["speedup"], name="toy speed-up", line_color=ACCENT),
                         go.Scatter(x=df["devices"], y=df["devices"], name="perfect scaling", line_dash="dot")])
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="devices",
                          yaxis_title="speed-up (simulated)")
        st.plotly_chart(fig, width="stretch")
