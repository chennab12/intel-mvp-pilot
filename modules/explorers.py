"""Intel Product Explorer and Software Stack Explorer."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.ui import content, page_header, render_sources, status_chip


def render_products() -> None:
    c = content()
    page_header(
        "🧩 Intel Product Explorer",
        "Products and families relevant to AI/ML Customer Engineering, with **only** the facts found in the cited "
        "sources and their check dates. This is a judgment aid, not a catalog: read the TPM notes and caveats.",
    )
    st.caption("Status key: 🟢 available · 🔵 announced · ⚪ historical · 🟠 deprecated · 🟡 unverified")
    products = c["products"]
    f1, f2, f3 = st.columns(3)
    cats = sorted({p["category"] for p in products})
    cat = f1.multiselect("Category", cats, key="pe_cat")
    statuses = sorted({p["status"] for p in products})
    stat = f2.multiselect("Status", statuses, key="pe_status")
    text = f3.text_input("Keyword", key="pe_text", placeholder="for example: HBM, NPU, PCIe")
    shown = [p for p in products
             if (not cat or p["category"] in cat) and (not stat or p["status"] in stat)
             and (not text or text.lower() in " ".join([p["name"], *p["key_facts"], p["tpm_notes"]]).lower())]
    st.dataframe(pd.DataFrame([{"Product": p["name"], "Category": p["category"], "Segment": p["segment"],
                                "Status": status_chip(p["status"])} for p in shown]),
                 hide_index=True, width="stretch")
    if not shown:
        st.info("No products match these filters.")
        return
    for p in shown:
        with st.expander(f"{status_chip(p['status'])} · {p['name']}"):
            st.markdown(f"**Category:** {p['category']} · **Segment:** {p['segment']} · **Example:** {p['example_sku']}")
            st.markdown("**Facts from sources**")
            for fact in p["key_facts"]:
                st.markdown(f"- {fact}")
            st.markdown(f"**Role in AI deployments:** {p['ai_role']}")
            st.markdown("**Software path:** " + " → ".join(p["software"]))
            st.info(f"**TPM notes:** {p['tpm_notes']}")
            st.warning(f"**Caveats / limits of comparison:** {p['caveats']}")
            render_sources(p["sources"])

    st.subheader("Codenames and naming: verify before mapping")
    st.markdown(
        "| Name you may hear | What the checked sources say | Status |\n|---|---|---|\n"
        "| B60 | Intel Arc Pro B60 datasheet (24 GB GDDR6) | 🟢 available (partner cards) |\n"
        "| B70 | Intel ARK: Arc Pro B70, launched Q1'26 | 🟢 available |\n"
        "| Crescent Island | Intel newsroom: data center inference GPU; Oct 2026 page names *Intel GPU D-series* | 🔵 announced |\n"
        "| Ponte Vecchio | PyTorch docs: Intel Data Center GPU Max Series | 🟡 lifecycle unverified |\n"
        "| Granite Rapids | Intel ARK: products formerly Granite Rapids (Xeon 6 P-core) | 🟢 available |\n"
        "| Battlemage | Intel ARK: products formerly Battlemage (Arc Pro B-Series) | 🟢 available |\n"
        "| Panther Lake | Core Ultra Series 3 (third-party + Intel newsroom) | 🟢 available; NPU TOPS unverified |"
    )
    st.caption("Before mapping any other codename to a commercial product, find it on Intel ARK or an Intel newsroom page.")


def render_stack() -> None:
    c = content()
    page_header(
        "🧱 Software Stack Explorer",
        "Trace a workload from the application down to the hardware. Each device type has its **own** stack; "
        "installation steps and supported software differ. Commands are shown for reading only; this app never runs them.",
    )
    names = {p["id"]: p["name"] for p in c["stack"]}
    pid = st.radio("Device path", list(names), format_func=names.get, horizontal=True, key="stack_pick")
    path = next(p for p in c["stack"] if p["id"] == pid)
    st.markdown(f"**Device type:** {path['device']}. Read top (your app) to bottom (hardware).")
    for i, layer in enumerate(path["layers"]):
        tag = ("confirmed in the cited source" if layer.get("verified")
               else "generic command, not from an official source here; needs verification for your version")
        with st.container(border=True):
            st.markdown(f"**{i + 1}. {layer['name']}**: {layer['examples']}")
            st.code(layer["check"], language="bash")
            st.caption(f"Check command: {tag}.")
            st.markdown(f"⚠️ *Typical failure:* {layer['fails_when']}")
            if layer.get("source"):
                src = c["sources"][layer["source"]]
                st.caption(f"Source: [{src['title']}]({src['url']}) · checked {src['checked'] or 'never (needs verification)'}")

    st.subheader("Compare the paths side by side")
    max_layers = max(len(p["layers"]) for p in c["stack"])
    table = {p["name"]: [layer["name"] + ": " + layer["examples"] for layer in p["layers"]] +
             [""] * (max_layers - len(p["layers"])) for p in c["stack"]}
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    st.caption("Notice: CPU, GPU, Gaudi and NPU paths share almost nothing below the framework layer.")
