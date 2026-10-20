"""Senior TPM Toolkit (templates), My Progress, and Glossary and Sources."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.ui import content, page_header, progress, status_chip
from utils.content import source_label, validate
from utils.progress import completion_pct, default_progress, export_json, import_json


def render_toolkit() -> None:
    c = content()
    page_header("🧰 Senior TPM Toolkit",
                "Editable templates prepopulated with **clearly fictional** examples. Edit in the box, then download as "
                "Markdown. Edits stay in this browser session only; download to keep them. Don't paste confidential "
                "customer or Intel information into a hosted copy of this app.")
    templates = c["templates"]
    names = {k: v["title"] for k, v in templates.items()}
    key = st.selectbox("Template", list(names), format_func=names.get, key="tk_pick")
    tpl = templates[key]
    state_key = f"tk_text_{key}"
    st.session_state.setdefault(state_key, tpl["body"])
    text = st.text_area("Edit (Markdown)", key=state_key, height=420)
    b1, b2 = st.columns(2)
    b1.download_button("⬇ Download .md", text, file_name=tpl["filename"], mime="text/markdown", type="primary")
    if b2.button("Reset to example"):
        st.session_state[state_key] = tpl["body"]
        st.rerun()
    with st.expander("Preview", expanded=False):
        st.markdown(text)


def render_progress() -> None:
    c = content()
    prog = progress()
    page_header("📊 My Progress",
                "Progress is kept in this browser session. **Export it as JSON to keep it**, and import it next time. "
                "Hosted Streamlit apps don't keep local files reliably (they can reset on restart), and this app "
                "deliberately has no shared progress file, so one learner's progress never mixes with another's.")
    all_ids = [lesson["id"] for lesson in c["lessons"]]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lessons complete", f"{len(prog['completed_lessons'])} / {len(all_ids)}")
    checks = prog["check_results"]
    right = sum(v["correct"] for v in checks.values())
    total = sum(v["total"] for v in checks.values())
    m2.metric("Knowledge checks", f"{right} / {total}" if total else "none yet")
    m3.metric("Labs done", f"{len(prog['labs_done'])} / {len(c['labs'])}")
    m4.metric("Scenarios played", f"{len(prog['scenarios'])} / {len(c['scenarios'])}")

    st.subheader("By track")
    for tid, track in c["tracks"].items():
        ids = [lesson["id"] for lesson in c["lessons"] if lesson["path"] == tid]
        pct = completion_pct(prog, ids)
        st.progress(pct / 100, text=f"{track['icon']} {track['name']}: {pct}%")
    st.subheader("By learning path")
    for path in c["learning_paths"]:
        pct = completion_pct(prog, path["lessons"])
        st.progress(pct / 100, text=f"{path['name']}: {pct}%")

    if prog.get("diagnostic"):
        d = prog["diagnostic"]
        st.info(f"Diagnostic: {d['correct']} / {d['total']} · recommended path: **{d['recommended_path']}**")
    if prog["scenarios"] or prog["cases"]:
        st.subheader("Simulator and clinic scores (best attempt)")
        rows = [{"type": "POC scenario", "name": next(s["title"] for s in c["scenarios"] if s["id"] == sid), **v}
                for sid, v in prog["scenarios"].items()]
        rows += [{"type": "Troubleshooting", "name": next(x["title"] for x in c["cases"] if x["id"] == cid),
                  "score": v["score"], "correct": v["correct"]} for cid, v in prog["cases"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.session_state.setdefault("prog_notes", prog.get("notes", ""))
    prog["notes"] = st.text_area("My notes (saved in the export)", key="prog_notes")

    st.subheader("Save and restore")
    st.download_button("⬇ Export progress (JSON)", export_json(prog), file_name="tpm_academy_progress.json",
                       mime="application/json", type="primary")
    up = st.file_uploader("Import progress (JSON exported from this app)", type=["json"], key="prog_upload")
    if up is not None and st.button("Load this file"):
        try:
            loaded, warnings = import_json(
                up.getvalue(), set(all_ids), {lab["id"] for lab in c["labs"]},
                {s["id"] for s in c["scenarios"]}, {x["id"] for x in c["cases"]})
        except ValueError as exc:
            st.error(f"Could not import: {exc}")
        else:
            st.session_state.progress = loaded
            st.session_state.prog_notes = loaded["notes"]
            st.success("Progress imported." + (f" {len(warnings)} unknown item(s) ignored." if warnings else ""))
            st.rerun()
    with st.expander("Reset progress"):
        if st.button("Reset all progress in this session"):
            st.session_state.progress = default_progress()
            st.session_state.prog_notes = ""
            st.rerun()
    st.caption("Completing lessons builds skills and evidence of learning. It does not guarantee promotion, which "
               "depends on sustained impact, scope and your organization's own process.")


def render_glossary() -> None:
    c = content()
    page_header("📚 Glossary and Sources", "Search the glossary, browse the source registry, and see how sources are "
                "kept current.")
    view = st.radio("Show", ["Glossary", "Source registry", "Verification workflow"], horizontal=True, key="gl_view")
    if view == "Glossary":
        terms = c["glossary"]
        a, b = st.columns([2, 1])
        q = a.text_input("Search terms", key="gl_q", placeholder="for example: KV, HBM, RoCE")
        cats = b.multiselect("Category", sorted({t["category"] for t in terms}), key="gl_cat")
        shown = [t for t in terms if (not cats or t["category"] in cats) and
                 (not q or q.lower() in (t["term"] + " " + t["definition"]).lower())]
        st.caption(f"{len(shown)} of {len(terms)} terms")
        st.dataframe(pd.DataFrame([{"Term": t["term"], "Category": t["category"], "Definition": t["definition"],
                                    "Source": c["sources"][t["source"]]["title"] if t.get("source") else ""}
                                   for t in sorted(shown, key=lambda t: t["term"].lower())]),
                     hide_index=True, width="stretch")
    elif view == "Source registry":
        sources = c["sources"]
        statuses = st.multiselect("Status", sorted({s["status"] for s in sources.values()}), key="src_status")
        rows = [{"ID": sid, "Title": s["title"], "Publisher": s["publisher"], "Kind": s["kind"],
                 "Status": status_chip(s["status"]), "Checked": source_label(s), "Review by": s["review_by"],
                 "Product": s["product"], "Version": s.get("version") or "", "URL": s["url"]}
                for sid, s in sources.items() if not statuses or s["status"] in statuses]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"URL": st.column_config.LinkColumn("URL", display_text="open")})
        pending = [sid for sid, s in sources.items() if s["status"] == "unverified" or not s["checked"]]
        st.warning(f"**Awaiting verification ({len(pending)}):** " + ", ".join(sources[s]["title"] for s in pending))
    else:
        st.markdown(
            "**How sources are kept honest**\n\n"
            "1. Every product-specific statement points at an entry in `content/sources.yaml`.\n"
            "2. Each entry records the URL, publisher, product, version, the facts used, the date it was **actually** "
            "opened (`checked`), a status, and a `review_by` date.\n"
            "3. If a page can't be opened or doesn't confirm a claim, `checked` stays `null` and status is `unverified`. "
            "The app then shows **Needs verification**. Dates are never filled in without a real check.\n"
            "4. The app never scrapes sources at page load, so it works offline and loads fast.\n"
            "5. **Monthly review:** filter the registry by `review_by` date, re-open each URL, update facts/version/date, "
            "and run `python -m pytest` (tests enforce the schema). See README → *Updating lessons and sources*.\n\n"
            "**Data labels used across the app:** measured · computed (arithmetic) · simulated · vendor claim · "
            "needs verification."
        )
        problems = validate()
        if problems:
            st.error("Content problems found:\n\n" + "\n".join(f"- {p}" for p in problems))
        else:
            st.success("Content schema check: all lessons, sources, scenarios and cases are valid.")
