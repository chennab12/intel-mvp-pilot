"""Senior TPM Concepts & Inference Math section: two cheatsheet tables."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from modules.ui import page_header
from utils import concepts_math as cm
from utils.content import source_label

TABLE_CSS = """
<style>
.cmw { overflow-x: auto; max-height: 720px; border: 1px solid rgba(128,128,128,.35); border-radius: 8px; margin-bottom: 1rem; }
.cmt { border-collapse: collapse; font-size: 0.86rem; }
.cmt th, .cmt td { border-bottom: 1px solid rgba(128,128,128,.25); padding: 8px 10px; vertical-align: top;
                   text-align: left; line-height: 1.45; }
.cmt th { position: sticky; top: 0; z-index: 2; background: #0068B5; color: #fff; font-weight: 600; white-space: nowrap; }
.cmt td.first { position: sticky; left: 0; z-index: 1; min-width: 200px; font-weight: 600; background: #FFFFFF;
                box-shadow: 2px 0 0 rgba(128,128,128,.25); }
.cmt tr.group td { background: #E6F0F8; font-weight: 700; font-size: .95rem; }
.cmt td.wide { min-width: 260px; } .cmt td.mid { min-width: 190px; }
.cmt code { white-space: pre-wrap; word-break: break-word; font-size: .8rem; }
.cmt .lv { white-space: nowrap; }
</style>
"""
WIDTH = {"plain": "wide", "why_tpm": "wide", "numbers": "mid", "behind": "wide", "formula": "wide",
         "example": "wide", "assumptions": "mid", "dependencies": "mid", "why": "mid", "note": "mid",
         "links": "mid", "depends_on": "mid"}


def _cell(row: dict, field: str) -> str:
    v = row[field]
    if field == "concept":
        return f"{html.escape(row['id'])}<br>{html.escape(v)}"
    if field == "step":
        return f"{row['icon']} {html.escape(row['id'])}<br>{html.escape(v)}"
    if field == "level":
        lv = cm.load()["levels"][v]
        return f"<span class='lv'>{lv['icon']} {lv['name']}</span>"
    if field == "depends_on":
        names = cm.concept_names()
        return "<br>".join(f"↳ {d} {html.escape(names[d])}" for d in v) or "—"
    if field == "links":
        srcs = cm.sources()
        return "<br>".join(f"<a href='{html.escape(srcs[x]['url'])}' target='_blank'>🔗&nbsp;{html.escape(srcs[x]['title'])}</a>"
                           for x in v) or "—"
    if field == "formula":
        return "<br>".join(f"<code>{html.escape(p.strip())}</code>" for p in str(v).split(" · "))
    if field in ("example", "numbers"):
        return "<br>".join(f"▸ {html.escape(p.strip())}" for p in str(v).split(" · "))
    if field == "practise":
        return f"🧪 {html.escape(v)}"
    return html.escape(str(v))


def _table(rows: list[dict], columns, group_field: str) -> str:
    head = "".join(f"<th>{h}</th>" for _, h in columns)
    body, current = [], None
    for r in rows:
        if r[group_field] != current:
            current = r[group_field]
            body.append(f"<tr class='group'><td class='first'>{html.escape(current)}</td><td colspan='{len(columns) - 1}'></td></tr>")
        cells = []
        for i, (f, _) in enumerate(columns):
            cls = "first" if i == 0 else WIDTH.get(f, "")
            cells.append(f"<td class='{cls}'>{_cell(r, f)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"{TABLE_CSS}<div class='cmw'><table class='cmt'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _match(row: dict, q: str) -> bool:
    return not q or q in " ".join(str(v) for v in row.values()).lower()


def render() -> None:
    d = cm.load()
    page_header("🎓 Senior TPM Concepts & Inference Math",
                "Two cheatsheet tables. **Table 1** lists the prerequisite concepts on the way to becoming a "
                "**senior, technically focused AI/ML TPM**, in learning order. **Table 2** walks the **math, statistics "
                "and formulas** behind testing whether **`Qwen/Qwen2.5-0.5B-Instruct` on vLLM** is functional and "
                "performant, with what happens behind the scenes at every step.")
    with st.expander("🧭 Legend, assumptions and dependencies", expanded=False):
        st.markdown(
            "**Levels:** 🟢 Foundation · 🟡 Working TPM · 🔴 Senior TPM. **Stages:** 🧰 setup · ✅ functional · "
            "⚡ performance · 📝 business.\n\n"
            "**Data labels:** *fact* = from a checked source (📚 links) · *computed* = arithmetic from those facts · "
            "*assumption* = an example value you must replace · ⚠️ pitfall · 💡 insight · ℹ️ note.\n\n"
            "**Global assumptions for Table 2:**\n"
            "- Model: Qwen2.5-0.5B-Instruct, 0.49 B parameters, 24 layers, 14 Q / 2 KV heads, head dim 64, BF16 weights and KV cache.\n"
            "- Device numbers: Intel Arc Pro B70, 32 GB, 608 GB/s, 230 W rated board power (Intel ARK). gpu-memory-utilization 0.9.\n"
            "- Speed numbers are **upper bounds** (perfect bandwidth use), not measurements. Prefill uses an **assumed** 10 TFLOPS achieved.\n"
            "- Cost/energy examples assume $1/h and 3,000 tok/s.\n\n"
            "**Dependencies:** vLLM (or Intel LLM-Scaler on the B70), a matching driver and runtime stack, the model's "
            "tokenizer and chat template, and a versioned eval set. Commands and flags change between versions; check your docs.")

    # ---------------- Table 1 ----------------
    st.subheader("📘 Table 1 · Prerequisite concepts for a senior technical AI/ML TPM")
    levels = d["levels"]
    a, b, c = st.columns([3, 3, 2])
    areas = list(dict.fromkeys(x["area"] for x in d["concepts"]))
    pick_areas = a.multiselect("Areas", areas, default=areas, key="cm_areas")
    pick_levels = b.multiselect("Levels", list(levels), default=list(levels), key="cm_levels",
                                format_func=lambda k: f"{levels[k]['icon']} {levels[k]['name']}")
    q1 = c.text_input("Filter", "", placeholder="KV, SLO, roofline...", key="cm_q1").lower().strip()
    concepts = [x for x in d["concepts"] if x["area"] in pick_areas and x["level"] in pick_levels and _match(x, q1)]
    counts = {k: sum(x["level"] == k for x in concepts) for k in levels}
    st.caption(f"{len(concepts)} of {len(d['concepts'])} concepts · " +
               " · ".join(f"{levels[k]['icon']} {counts[k]}" for k in levels) +
               ". Rows are in learning order: everything under 'Learn first' appears earlier in the table.")
    if concepts:
        st.markdown(_table(concepts, cm.CONCEPT_COLUMNS, "area"), unsafe_allow_html=True)
    else:
        st.warning("No concepts match. Clear a filter.")

    # ---------------- Table 2 ----------------
    st.subheader("🧮 Table 2 · Inference math: Qwen2.5-0.5B-Instruct on vLLM, functional → performance")
    st.caption("Read it top to bottom as the life of a test: set up memory, prove it's correct, then prove it's fast "
               "and affordable. Every worked number is computed from public specs or a stated assumption.")
    stages = list(dict.fromkeys(x["stage"] for x in d["math"]))
    a, b = st.columns([3, 2])
    pick_stages = a.multiselect("Stages", stages, default=stages, key="cm_stages")
    q2 = b.text_input("Filter", "", placeholder="TTFT, percentile, KV...", key="cm_q2").lower().strip()
    math_rows = [x for x in d["math"] if x["stage"] in pick_stages and _match(x, q2)]
    st.caption(f"{len(math_rows)} of {len(d['math'])} steps.")
    if math_rows:
        st.markdown(_table(math_rows, cm.MATH_COLUMNS, "stage"), unsafe_allow_html=True)
    else:
        st.warning("No steps match. Clear a filter.")

    st.markdown("**🧠 The intuition in four lines**\n"
                "1. 🧰 Memory first: weights are a fixed cost, the KV cache is the per-user cost, and together they cap concurrency.\n"
                "2. ✅ Correctness is statistical: decide tolerances and sample sizes before you compare.\n"
                "3. ⚡ Prefill is math-bound (TTFT); decode is memory-bound (TPOT); batching shares the weight reads.\n"
                "4. 📝 Judge at an SLO: goodput, then $/1M tokens and J/token at that load.")

    a, b, c = st.columns(3)
    a.download_button("⬇️ Both tables (Markdown)", cm.to_markdown(concepts, math_rows),
                      file_name="tpm_concepts_and_inference_math.md", mime="text/markdown", key="cm_md")
    b.download_button("⬇️ Table 1 (CSV)", cm.to_csv(concepts, cm.CONCEPT_COLUMNS), file_name="tpm_concepts.csv",
                      mime="text/csv", key="cm_csv1")
    c.download_button("⬇️ Table 2 (CSV)", cm.to_csv(math_rows, cm.MATH_COLUMNS), file_name="inference_math.csv",
                      mime="text/csv", key="cm_csv2")

    st.subheader("📚 Sources and verification")
    srcs = cm.sources()
    ids = list(dict.fromkeys(x for r in d["concepts"] + d["math"] for x in r["links"]))
    st.dataframe(pd.DataFrame([{"Source": srcs[i]["title"], "Publisher": srcs[i].get("publisher", ""),
                                "Checked": source_label(srcs[i]), "URL": srcs[i]["url"]}
                               for i in ids]), hide_index=True, width="stretch",
                 column_config={"URL": st.column_config.LinkColumn("URL", display_text="open")})
