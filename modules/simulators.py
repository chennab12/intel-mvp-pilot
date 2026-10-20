"""Customer POC Simulator and Troubleshooting Clinic."""
from __future__ import annotations

import streamlit as st

from modules.ui import content, goto, page_header, progress, render_sources
from utils.poc import choose, new_run, score_case, score_run


def render_poc() -> None:
    c = content()
    page_header("🧭 Customer POC Simulator",
                "Branching scenarios with fictional customers. Your decisions change what evidence you collect and what "
                "happens next. You are scored on **reasoning** (choosing the decision-driving step) and "
                "**completeness** (collecting the evidence a sound recommendation needs), not memorized answers.")
    names = {s["id"]: s["title"] for s in c["scenarios"]}
    done = progress()["scenarios"]
    sid = st.selectbox("Scenario", list(names), key="poc_pick",
                       format_func=lambda i: names[i] + (f"  ·  best score {done[i]['score']:.0f}%" if i in done else ""))
    sc = next(s for s in c["scenarios"] if s["id"] == sid)
    key = f"poc_run_{sid}"
    if key not in st.session_state:
        st.session_state[key] = new_run(sc)
    run = st.session_state[key]

    st.markdown(f"**Customer:** {sc['customer']} · **Skills:** {', '.join(sc['skills'])}")
    st.info(sc["brief"])

    left, right = st.columns([2, 1])
    with right:
        st.markdown("**Evidence collected**")
        for ev_key, desc in sc["required_evidence"].items():
            st.markdown(("✅ " if ev_key in run["evidence"] else "⬜ ") + desc)
        st.progress(len([e for e in run["evidence"] if e in sc["required_evidence"]]) / len(sc["required_evidence"]))
        if st.button("Restart scenario", key=f"restart_{sid}"):
            st.session_state[key] = new_run(sc)
            st.rerun()

    with left:
        for step in run["history"]:
            node = sc["nodes"][step["node"]]
            with st.container(border=True):
                st.markdown(f"*{node['text']}*")
                st.markdown(f"➡️ **You chose:** {step['label']}")
                (st.success if step["points"] >= 2 else st.warning if step["points"] == 1 else st.error)(step["feedback"])
        if not run["done"]:
            node = sc["nodes"][run["node"]]
            st.markdown(f"**{node['text']}**")
            for i, choice in enumerate(node["choices"]):
                if st.button(choice["label"], key=f"{sid}_{run['node']}_{i}_{len(run['history'])}"):
                    st.session_state[key] = choose(sc, run, i)
                    st.rerun()
        else:
            ending = sc["endings"][run["node"]]
            st.markdown("#### Outcome")
            st.markdown(ending["outcome"])
            result = score_run(sc, run)
            m1, m2, m3 = st.columns(3)
            m1.metric("Reasoning", f"{result['reasoning']}%")
            m2.metric("Completeness", f"{result['completeness']}%")
            m3.metric("Overall", f"{result['score']}%")
            if result["missing"]:
                st.warning("Evidence you didn't collect: " + "; ".join(sc["required_evidence"][m] for m in result["missing"]))
            else:
                st.success("You collected all the evidence a sound recommendation needs.")
            best = done.get(sid)
            if not best or result["score"] > best["score"]:
                done[sid] = {k: result[k] for k in ("reasoning", "completeness", "score")}
            st.caption("Try the scenario again with different choices to see other outcomes.")


def render_clinic() -> None:
    c = content()
    page_header("🩺 Troubleshooting Clinic",
                "Seven fictional cases. **Gather evidence before choosing a fix.** Each evidence action reveals an "
                "observation (illustrative, written for teaching, not real command output). Scoring: 50% key evidence, "
                "40% correct diagnosis (half credit if you diagnose with too little evidence), 10% for avoiding premature fixes.")
    names = {case["id"]: case["title"] for case in c["cases"]}
    done = progress()["cases"]
    cid = st.selectbox("Case", list(names), key="case_pick",
                       format_func=lambda i: names[i] + (f"  ·  best {done[i]['score']:.0f}%" if i in done else ""))
    case = next(x for x in c["cases"] if x["id"] == cid)
    gkey, dkey = f"case_gathered_{cid}", f"case_diag_{cid}"
    st.session_state.setdefault(gkey, [])
    gathered = st.session_state[gkey]

    st.markdown(f"**Platform:** {case['platform']}")
    st.error(f"**Symptom:** {case['symptom']}")

    st.markdown("#### Actions")
    for action in case["actions"]:
        taken = action["id"] in gathered
        icon = "🔎" if action["kind"] == "evidence" else "🔧"
        if taken:
            with st.container(border=True):
                st.markdown(f"{icon} **{action['label']}**")
                st.code(action["observation"], language="text", wrap_lines=True)
                st.caption("Illustrative observation (simulated).")
        elif st.button(f"{icon} {action['label']}", key=f"{cid}_{action['id']}", disabled=st.session_state.get(dkey) is not None):
            gathered.append(action["id"])
            st.rerun()

    st.markdown("#### Your diagnosis")
    if st.session_state.get(dkey) is None:
        pick = st.radio("Most likely cause", list(range(len(case["diagnoses"]))), index=None,
                        format_func=lambda i: case["diagnoses"][i]["label"], key=f"{cid}_diag_radio")
        if st.button("Submit diagnosis", type="primary", disabled=pick is None, key=f"{cid}_submit"):
            st.session_state[dkey] = pick
            st.rerun()
    else:
        pick = st.session_state[dkey]
        res = score_case(case, gathered, pick)
        (st.success if res["correct"] else st.error)(
            f"You chose: {case['diagnoses'][pick]['label']} - {'correct' if res['correct'] else 'not the best-supported cause'}.")
        m1, m2 = st.columns(2)
        m1.metric("Score", f"{res['score']}%")
        m2.metric("Key evidence found", f"{len(res['key_found'])} / {len(res['key_found']) + len(res['key_missing'])}")
        if res["key_missing"]:
            labels = {a["id"]: a["label"] for a in case["actions"]}
            st.warning("Key evidence you skipped: " + "; ".join(labels[k] for k in res["key_missing"]))
        if res["premature"]:
            st.warning("You tried a fix before having evidence. In a real POC that costs time and trust.")
        st.markdown(f"**Resolution:** {case['resolution']}")
        best = done.get(cid)
        if not best or res["score"] > best["score"]:
            done[cid] = {"score": res["score"], "correct": res["correct"]}
        render_sources(case["sources"], heading="Related sources")
        b1, b2 = st.columns(2)
        if b1.button("Reset case", key=f"{cid}_reset"):
            st.session_state[gkey] = []
            st.session_state[dkey] = None
            st.rerun()
        b2.button("Open related lesson", key=f"{cid}_lesson", on_click=goto, args=("Learning Paths",),
                  kwargs={"lesson_request": case["lesson"], "lp_view": "Lessons"})
