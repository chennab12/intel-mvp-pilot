"""Golden Master Cheatsheet section: one end-to-end table from 'is it functional?' to 'is it performant?'."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from modules.ui import page_header
from utils import cheatsheet as cs

CODE_FIELDS = {"hf_cmd", "vllm_cmd", "b70_cmd"}
CODE_TABS = [("hf_code", "hf_cmd", "🐍 Transformers + PyTorch (CPU)"), ("vllm_code", "vllm_cmd", "🚀 vLLM"),
             ("b70_code", "b70_cmd", "🔷 Intel Arc Pro B70")]

TABLE_CSS = """
<style>
.gm-wrap { overflow-x: auto; border: 1px solid rgba(128,128,128,.35); border-radius: 8px; }
.gm { border-collapse: collapse; font-size: 0.86rem; min-width: 1500px; }
.gm th, .gm td { border-bottom: 1px solid rgba(128,128,128,.25); padding: 8px 10px; vertical-align: top;
                 text-align: left; line-height: 1.45; }
.gm th { position: sticky; top: 0; background: #0068B5; color: #fff; font-weight: 600; white-space: nowrap; }
.gm td.step { position: sticky; left: 0; min-width: 190px; font-weight: 600;
              background: #FFFFFF; box-shadow: 2px 0 0 rgba(128,128,128,.25); }
.gm tr.phase td, .gm tr.phase td.step { background: #E6F0F8; font-weight: 700; font-size: .95rem; }
.gm code { white-space: pre-wrap; word-break: break-word; font-size: .8rem; }
.gm td.why { min-width: 200px; } .gm td.cmd { min-width: 230px; } .gm td.metrics { min-width: 260px; }
.gm td.links { min-width: 220px; }
.gm .kw { display: inline-block; border: 1px solid rgba(0,104,181,.5); border-radius: 10px; padding: 0 6px;
          margin: 1px 2px; font-size: .75rem; }
</style>
"""
CLASS = {"step": "step", "why_simple": "why", "hf_cmd": "cmd", "vllm_cmd": "cmd", "b70_cmd": "cmd",
         "b70_why": "why", "metrics": "metrics", "links": "links"}


def _html_cell(step: dict, field: str) -> str:
    v = step[field]
    if field == "step":
        return f"{step['icon']} {html.escape(step['id'])}<br>{html.escape(v)}"
    if field in CODE_FIELDS:
        return f"<code>{html.escape(v)}</code>"
    if field == "metrics":
        return "<br>".join(f"▸ {html.escape(x)}" for x in v)
    if field == "keywords":
        return " ".join(f"<span class='kw'>{html.escape(x)}</span>" for x in v)
    if field == "links":
        srcs = cs.sources()
        return "<br>".join(f"<a href='{html.escape(srcs[x]['url'])}' target='_blank'>🔗&nbsp;{html.escape(srcs[x]['title'])}</a>"
                           for x in v) or "—"
    return html.escape(str(v))


def _table(steps: list[dict]) -> str:
    phases = {p["id"]: p for p in cs.load()["phases"]}
    head = "".join(f"<th>{h}</th>" for _, h in cs.COLUMNS)
    rows, current = [], None
    for s in steps:
        if s["phase"] != current:
            current = s["phase"]
            p = phases[current]
            rows.append(f"<tr class='phase'><td class='step' colspan='1'>{p['icon']} {html.escape(p['name'])}</td>"
                        f"<td colspan='{len(cs.COLUMNS) - 1}'></td></tr>")
        rows.append("<tr>" + "".join(f"<td class='{CLASS.get(f, '')}'>{_html_cell(s, f)}</td>" for f, _ in cs.COLUMNS)
                    + "</tr>")
    return f"{TABLE_CSS}<div class='gm-wrap'><table class='gm'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def render() -> None:
    d = cs.load()
    m, b = d["model"], d["b70"]
    page_header("🧾 Golden Master Cheatsheet",
                f"One table, end to end: first prove **`{m['id']}`** is **functional ✅**, then prove it is "
                "**performant ⚡**. For every step you get the command for **Hugging Face Transformers + PyTorch "
                "(CPU)**, the **vLLM** equivalent, what changes on an **Intel Arc Pro B70** GPU and why, the key "
                "metrics and formulas, and why the step matters, in plain words.")
    st.info("📌 **Commands are illustrative.** This app does not run them. Check each flag against the docs for "
            "your exact versions, on a system you're authorized to use. The Arc Pro B70 is a GPU (Intel lists it "
            "as \"products formerly Battlemage\"); it runs alongside a host CPU.")
    with st.expander("🧭 Legend: icons and data labels", expanded=False):
        st.markdown("🧰 setup · ✅ functional · ⚡ performance · 📝 report · 🐍 Transformers + PyTorch · 🚀 vLLM · "
                    "🔷 Intel Arc Pro B70 · ❓ why it differs · 📐 metrics and formulas · 🔑 core concepts · 💡 why "
                    "this step · 🔗 source\n\n"
                    "**Fact** = from a checked source (see Sources below). **Computed** = arithmetic from those facts. "
                    "**Assumption** = an example value you replace with your own. **Illustrative** = a command "
                    "shown, never run here.")

    st.subheader("🔢 Key numbers for this model (computed)")
    k = cs.key_numbers()
    c = st.columns(4)
    c[0].metric("Parameters", f"{m['params_billion']} B", help=f"{m['non_embedding_params_billion']} B non-embedding (model card)")
    c[1].metric("BF16 weights", f"{k['bf16_weights_gb']:.2f} GB", help="params × 2 bytes")
    c[2].metric("KV cache per token", f"{k['kv_bytes_per_token'] / 1024:.0f} KiB",
                help=f"2 × {m['layers']} layers × {m['kv_heads']} KV heads × {m['head_dim']} head dim × 2 bytes")
    c[3].metric("KV per 32K-token sequence", f"{k['kv_gb_per_full_context']:.2f} GB")
    c = st.columns(4)
    c[0].metric(f"B70 memory ({b['bandwidth_gb_s']} GB/s)", f"{b['memory_gb']} GB", help="Intel ARK: GDDR6, 256-bit")
    c[1].metric("B70 decode ceiling (batch 1)", f"≈ {k['b70_decode_ceiling_tok_s']:.0f} tok/s",
                help="bandwidth ÷ weight bytes: an upper bound, not a measurement")
    c[2].metric("Laptop decode ceiling", f"≈ {k['laptop_decode_ceiling_tok_s']:.0f} tok/s",
                help=f"assumes {d['assumptions']['laptop_bandwidth_gb_s']} GB/s; check your own spec")
    c[3].metric("Full-context sequences on B70", f"≈ {k['b70_full_context_sequences']}",
                help=f"(32 GB × {d['assumptions']['gpu_memory_utilization']} − weights) ÷ KV per sequence; before activations")
    st.caption(f"Architecture: {m['architecture']}. Context {m['context_tokens']:,} tokens, generation "
               f"{m['generation_tokens']:,} tokens, {m['license']}. Rule of thumb: **decode is memory-bandwidth "
               "bound, prefill is compute bound**.")

    st.subheader("🗂️ Master table")
    a, bcol, ccol = st.columns([3, 2, 2])
    phase_names = {p["id"]: f"{p['icon']} {p['name']}" for p in d["phases"]}
    chosen = a.multiselect("Phases", list(phase_names), default=list(phase_names), format_func=phase_names.get,
                           key="gm_phases")
    query = bcol.text_input("Filter steps", "", placeholder="TTFT, KV cache, xpu...", key="gm_query")
    view = ccol.radio("View", ["Table", "Step cards"], horizontal=True, key="gm_view")
    steps = cs.filtered_steps(chosen, query)
    st.caption(f"{len(steps)} of {len(d['steps'])} steps. Scroll the table sideways; the step column stays put.")
    if not steps:
        st.warning("No steps match. Clear the filter.")
    elif view == "Table":
        st.markdown(_table(steps), unsafe_allow_html=True)
    else:
        for s in steps:
            with st.expander(f"{s['icon']} {s['id']} · {s['step']}"):
                st.markdown(f"💡 **Why:** {s['why_simple']}")
                tabs = st.tabs([t for _, _, t in CODE_TABS])
                for tab, (code_field, cmd_field, _) in zip(tabs, CODE_TABS):
                    with tab:
                        lang = "python" if "import " in s.get(code_field, "") or "(" in s[cmd_field][:40] else "bash"
                        st.code(s.get(code_field) or s[cmd_field], language=lang, wrap_lines=True)
                st.markdown(f"❓ **Why it differs on B70:** {s['b70_why']}")
                st.markdown("📐 **Key metrics:**\n" + "\n".join(f"- {x}" for x in s["metrics"]))
                st.markdown("🔑 " + " · ".join(f"`{x}`" for x in s["keywords"]))
                if s["links"]:
                    srcs = cs.sources()
                    st.markdown(" · ".join(f"🔗 [{srcs[x]['title']}]({srcs[x]['url']})" for x in s["links"]))

    a, bcol = st.columns(2)
    a.download_button("⬇️ Download as Markdown", cs.to_markdown(steps), file_name="golden_master_cheatsheet.md",
                      mime="text/markdown", key="gm_md", disabled=not steps)
    bcol.download_button("⬇️ Download as CSV", cs.to_csv(steps), file_name="golden_master_cheatsheet.csv",
                         mime="text/csv", key="gm_csv", disabled=not steps)

    st.subheader("🧠 The whole flow in one line")
    st.markdown("**Hardware → driver → framework sees device → right model → loads → one reply → same reply twice → "
                "known answers → matches reference → passes the accuracy bar → survives edge cases → API keeps its "
                "promises** ✅ … then ⚡ **warm up → latency (TTFT/TPOT) → memory budget → throughput sweep → "
                "goodput vs SLO → precision trade-off → cost/energy → reproducible report** 📝")

    st.subheader("🔗 Sources and verification")
    ids = list(dict.fromkeys(x for s in d["steps"] for x in s["links"]))
    st.dataframe(pd.DataFrame(cs.source_rows(ids)), hide_index=True, width="stretch",
                 column_config={"URL": st.column_config.LinkColumn("URL", display_text="open")})
    with st.expander("Facts this cheatsheet takes from these sources"):
        srcs = cs.sources()
        for sid in ids:
            st.markdown(f"**{srcs[sid]['title']}**")
            for fact in srcs[sid]["facts"]:
                st.markdown(f"- {fact}")
