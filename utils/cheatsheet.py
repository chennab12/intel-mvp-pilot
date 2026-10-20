"""Golden Master Cheatsheet: loading, computed key numbers, validation and Markdown/CSV export.

Content lives in content/cheatsheet.yaml. No Streamlit imports here so everything is unit-testable.
"""
from __future__ import annotations

import csv
import io
from functools import lru_cache
from pathlib import Path

import yaml

from utils import calculators as calc
from utils.content import load_all, source_label

PATH = Path(__file__).resolve().parent.parent / "content" / "cheatsheet.yaml"
STEP_FIELDS = ["id", "phase", "icon", "step", "why_simple", "hf_cmd", "vllm_cmd", "b70_cmd", "b70_why",
               "metrics", "keywords", "links"]
COLUMNS = [  # (field, header) in table order
    ("step", "🪜 Step"),
    ("why_simple", "💡 Why (in simple terms)"),
    ("hf_cmd", "🐍 Transformers + PyTorch (CPU)"),
    ("vllm_cmd", "🚀 vLLM"),
    ("b70_cmd", "🔷 On Intel Arc Pro B70"),
    ("b70_why", "❓ Why it differs on B70"),
    ("metrics", "📐 Key metrics · numbers · formulas"),
    ("keywords", "🔑 Core concepts"),
    ("links", "🔗 Links"),
]


@lru_cache(maxsize=1)
def load() -> dict:
    with open(PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sources() -> dict:
    """Cheatsheet sources first, then the app's main registry."""
    return {**load_all()["sources"], **load()["sources"]}


def key_numbers() -> dict:
    """Arithmetic from the model card / config and the B70 spec. All values are COMPUTED, not measured."""
    d = load()
    m, g, a = d["model"], d["b70"], d["assumptions"]
    weights = calc.weight_memory_bytes(m["params_billion"], "BF16")
    kv_token = calc.kv_cache_bytes(m["layers"], m["kv_heads"], m["head_dim"], 2, 1, 1)
    kv_full = kv_token * m["context_tokens"]
    usable = g["memory_gb"] * a["gpu_memory_utilization"] * calc.GB
    kv_tokens_b70 = (usable - weights) / kv_token
    return {
        "weights_gb": {p: calc.weight_memory_bytes(m["params_billion"], p) / calc.GB for p in calc.BYTES_PER_PARAM},
        "bf16_weights_gb": weights / calc.GB,
        "kv_bytes_per_token": kv_token,
        "kv_gb_per_full_context": kv_full / calc.GB,
        "b70_decode_ceiling_tok_s": g["bandwidth_gb_s"] * calc.GB / weights,
        "laptop_decode_ceiling_tok_s": a["laptop_bandwidth_gb_s"] * calc.GB / weights,
        "b70_kv_tokens": kv_tokens_b70,
        "b70_full_context_sequences": int(kv_tokens_b70 // m["context_tokens"]),
        "flops_per_token": 2 * m["params_billion"] * 1e9,
        "head_dim_check": m["hidden_size"] / m["q_heads"],
    }


def validate() -> list[str]:
    """Schema problems in the cheatsheet content (empty list = OK)."""
    d = load()
    problems = []
    phases = {p["id"] for p in d["phases"]}
    srcs = sources()
    ids = set()
    for s in d["steps"]:
        for f in STEP_FIELDS:
            if f not in s or s[f] in (None, ""):
                problems.append(f"{s.get('id', '?')}: missing {f}")
        if s.get("id") in ids:
            problems.append(f"duplicate id {s['id']}")
        ids.add(s.get("id"))
        if s.get("phase") not in phases:
            problems.append(f"{s.get('id')}: unknown phase {s.get('phase')}")
        for link in s.get("links", []):
            if link not in srcs:
                problems.append(f"{s.get('id')}: unknown source {link}")
    for sid, src in d["sources"].items():
        for f in ("title", "url", "kind", "status", "facts", "checked"):
            if f not in src:
                problems.append(f"source {sid}: missing {f}")
        if src.get("kind") == "third-party" and src.get("status") != "unverified":
            problems.append(f"source {sid}: third-party sources must be unverified")
    return problems


def filtered_steps(phases: list[str] | None = None, query: str = "") -> list[dict]:
    q = query.lower().strip()
    out = []
    for s in load()["steps"]:
        if phases and s["phase"] not in phases:
            continue
        if q and q not in " ".join(str(v) for v in s.values()).lower():
            continue
        out.append(s)
    return out


def _cell(step: dict, field: str, link_fmt) -> str:
    v = step[field]
    if field == "step":
        return f"{step['icon']} {step['id']} · {v}"
    if field == "links":
        srcs = sources()
        return " · ".join(link_fmt(srcs[x]["title"], srcs[x]["url"]) for x in v) or "—"
    if isinstance(v, list):
        return " • ".join(v)
    return str(v)


def to_markdown(steps: list[dict]) -> str:
    d = load()
    m = d["model"]
    lines = [f"# Golden Master Cheatsheet: {m['id']}", "",
             "Functional first, then performance. Transformers + PyTorch on CPU, vLLM, and Intel Arc Pro B70. "
             "Commands are illustrative: check flags for your exact versions. Computed numbers are arithmetic, "
             "not measurements.", "",
             "| # | " + " | ".join(h for _, h in COLUMNS) + " |",
             "|---|" + "---|" * len(COLUMNS)]
    for i, s in enumerate(steps, 1):
        cells = [_cell(s, f, lambda t, u: f"[{t}]({u})").replace("|", "\\|").replace("\n", " ") for f, _ in COLUMNS]
        lines.append(f"| {i} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def to_csv(steps: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "phase"] + [h for _, h in COLUMNS])
    for s in steps:
        w.writerow([s["id"], s["phase"]] + [_cell(s, f, lambda t, u: f"{t} <{u}>") for f, _ in COLUMNS])
    return buf.getvalue()


def source_rows(ids: list[str]) -> list[dict]:
    srcs = sources()
    return [{"Source": srcs[i]["title"], "Publisher": srcs[i].get("publisher", ""), "Checked": source_label(srcs[i]),
             "URL": srcs[i]["url"]} for i in ids]
