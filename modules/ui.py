"""Shared Streamlit helpers: badges, visuals, source boxes and progress access."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.content import load_all, source_label
from utils.progress import default_progress

LABEL_ICONS = {
    "Runs on a normal laptop": "💻",
    "Simulation only": "🧪",
    "Requires specified Intel hardware": "🔌",
}
STATUS_ICONS = {
    "available": "🟢", "announced": "🔵", "historical": "⚪", "deprecated": "🟠", "unverified": "🟡", "concept": "📘",
}
KIND_TEXT = {
    "measured": "**Measured** just now on the machine running this app.",
    "computed": "**Computed** from fixed inputs (arithmetic, not a hardware measurement).",
    "simulated": "**Simulated** data from a toy model. Not a measurement of any product.",
}
ACCENT = "#0068B5"


def content() -> dict:
    return load_all()


def progress() -> dict:
    if "progress" not in st.session_state:
        st.session_state.progress = default_progress()
    return st.session_state.progress


def goto(section: str, **targets) -> None:
    """Callback for buttons that jump to another section (and optionally an item in it)."""
    st.session_state.nav = section
    for key, value in targets.items():
        st.session_state[key] = value


def label_badge(label: str, hardware: str | None = None) -> None:
    icon = LABEL_ICONS.get(label, "•")
    text = f"{icon} **{label}**"
    if hardware:
        text += f": {hardware}"
    if label == "Runs on a normal laptop":
        st.success(text)
    elif label == "Simulation only":
        st.info(text)
    else:
        st.warning(text)


def status_chip(status: str) -> str:
    return f"{STATUS_ICONS.get(status, '•')} {status}"


def render_visual(visual: dict) -> None:
    caption = visual.get("caption", "")
    if visual["type"] == "table":
        df = pd.DataFrame(visual["rows"], columns=visual["columns"])
        st.dataframe(df, hide_index=True, width="stretch")
    elif visual["type"] == "dot":
        st.graphviz_chart(visual["source"], width="stretch")
    elif visual["type"] == "bars":
        fig = go.Figure(go.Bar(x=visual["x"], y=visual["y"], marker_color=ACCENT,
                               text=[f"{v:g}" for v in visual["y"]], textposition="outside"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title=visual.get("x_label"), yaxis_title=visual.get("y_label"))
        st.plotly_chart(fig, width="stretch")
    if caption:
        st.caption(caption)


def render_sources(source_ids: list[str], heading: str = "Sources and verification") -> None:
    sources = content()["sources"]
    st.markdown(f"**{heading}**")
    rows = []
    for sid in source_ids:
        src = sources[sid]
        rows.append({
            "Source": src["title"],
            "Status": status_chip(src["status"]),
            "Checked": source_label(src),
            "URL": src["url"],
            "Product / family": src["product"],
            "Version": src.get("version") or "n/a",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"URL": st.column_config.LinkColumn("URL", display_text="open")})
    with st.expander("Facts this app takes from these sources"):
        for sid in source_ids:
            src = sources[sid]
            st.markdown(f"**{src['title']}** ({src['publisher']}, {src['kind']})")
            for fact in src["facts"]:
                st.markdown(f"- {fact}")


def render_result(result: dict) -> None:
    """Show a lab runner result with its data label."""
    st.markdown(KIND_TEXT.get(result.get("kind"), ""))
    metrics = result.get("metrics") or {}
    if metrics:
        cols = st.columns(min(3, len(metrics)))
        for i, (k, v) in enumerate(metrics.items()):
            cols[i % len(cols)].metric(k, str(v))
    if result.get("table"):
        st.dataframe(pd.DataFrame(result["table"]), hide_index=True, width="stretch")
    for note in result.get("notes", []):
        st.caption(note)


def page_header(title: str, intro: str) -> None:
    st.title(title)
    st.markdown(intro)
