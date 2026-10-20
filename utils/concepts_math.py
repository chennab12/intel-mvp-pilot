"""Senior TPM Concepts & Inference Math: loading, validation, computed checks and export.

Content lives in content/concepts_math.yaml. No Streamlit imports here so everything is unit-testable.
"""
from __future__ import annotations

import csv
import io
import math
from functools import lru_cache
from pathlib import Path

import yaml

from utils import calculators as calc
from utils import cheatsheet as cs

PATH = Path(__file__).resolve().parent.parent / "content" / "concepts_math.yaml"

CONCEPT_COLUMNS = [
    ("concept", "🧩 Concept"), ("level", "📶 Level"), ("plain", "💡 What it is (plain words)"),
    ("why_tpm", "🎯 Why a senior technical TPM needs it"), ("numbers", "🔢 Key numbers · rules of thumb"),
    ("depends_on", "🔗 Learn first"), ("practise", "🧪 Practise in this app"), ("links", "📚 Sources"),
]
MATH_COLUMNS = [
    ("step", "🪜 Step"), ("behind", "🔍 What happens behind the scenes"), ("formula", "📐 Formula"),
    ("example", "🧮 Worked example (Qwen2.5-0.5B on vLLM)"), ("metric", "📏 Metric · unit"),
    ("assumptions", "📌 Assumptions"), ("dependencies", "🔗 Depends on"), ("why", "💡 Why this step"),
    ("note", "📝 Annotation"), ("links", "📚 Sources"),
]
CONCEPT_FIELDS = ["id", "area", "concept", "level", "plain", "why_tpm", "numbers", "depends_on", "practise", "links"]
MATH_FIELDS = ["id", "stage", "icon", "step", "behind", "formula", "example", "metric", "assumptions",
               "dependencies", "why", "note", "links"]


@lru_cache(maxsize=1)
def load() -> dict:
    with open(PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sources() -> dict:
    return {**cs.sources(), **load()["sources"]}


def concept_names() -> dict:
    return {c["id"]: c["concept"] for c in load()["concepts"]}


def validate(section_names: list[str] | None = None) -> list[str]:
    d = load()
    srcs = sources()
    ids = set(concept_names())
    problems = []
    for kind, rows, fields in (("concept", d["concepts"], CONCEPT_FIELDS), ("math", d["math"], MATH_FIELDS)):
        seen = set()
        for r in rows:
            for f in fields:
                if f not in r or (r[f] in (None, "") and f not in ("depends_on", "links")):
                    problems.append(f"{kind} {r.get('id', '?')}: missing {f}")
            if r.get("id") in seen:
                problems.append(f"duplicate {kind} id {r['id']}")
            seen.add(r.get("id"))
            for link in r.get("links") or []:
                if link not in srcs:
                    problems.append(f"{r.get('id')}: unknown source {link}")
    order = [c["id"] for c in d["concepts"]]
    for c in d["concepts"]:
        if c["level"] not in d["levels"]:
            problems.append(f"{c['id']}: unknown level {c['level']}")
        for dep in c["depends_on"]:
            if dep not in ids:
                problems.append(f"{c['id']}: unknown prerequisite {dep}")
            elif order.index(dep) >= order.index(c["id"]):
                problems.append(f"{c['id']}: prerequisite {dep} must come earlier")
        if section_names is not None and c["practise"] not in section_names:
            problems.append(f"{c['id']}: unknown app section {c['practise']}")
    return problems


def worked_numbers() -> dict:
    """Recompute the worked examples quoted in Table 2 (all COMPUTED from facts or stated assumptions)."""
    m, b = cs.load()["model"], cs.load()["b70"]
    w = calc.weight_memory_bytes(m["params_billion"], "BF16")
    kv = calc.kv_cache_bytes(m["layers"], m["kv_heads"], m["head_dim"], 2, 1, 1)
    bw = b["bandwidth_gb_s"] * calc.GB
    kv_space = b["memory_gb"] * 0.9 * calc.GB - w
    step1 = (w + kv * 1024) / bw
    step32 = (w + 32 * kv * 1024) / bw
    se = math.sqrt(0.8 * 0.2 / 200)
    return {
        "weights_gb": w / calc.GB,
        "kv_bytes_per_token": kv,
        "kv_mb_4k": kv * 4096 / 1e6,
        "kv_gb_32k": kv * m["context_tokens"] / calc.GB,
        "kv_space_gb": kv_space / calc.GB,
        "kv_tokens_m": kv_space / kv / 1e6,
        "blocks_16": kv_space / (16 * kv),
        "blocks_64": kv_space / (64 * kv),
        "max_4k_requests": int(kv_space / kv // 4096),
        "flops_per_token_g": 2 * m["params_billion"],
        "attention_mflop_4k": 4 * m["layers"] * 4096 * m["hidden_size"] / 1e6,
        "prefill_tflop_1k": 2 * m["params_billion"] * 1e9 * 1024 / 1e12,
        "prefill_ms_at_10tflops": 2 * m["params_billion"] * 1e9 * 1024 / 10e12 * 1e3,
        "decode_ms_b1": step1 * 1e3,
        "decode_tok_s_b1": 1 / step1,
        "batch32_ms": step32 * 1e3,
        "batch32_tok_s": 32 / step32,
        "batch32_per_user": 1 / step32,
        "e2e_s": 0.1 + 255 * 0.005,
        "little_l": 10 * (0.1 + 255 * 0.005),
        "acc_se_pts": se * 100,
        "acc_ci_pts": 1.96 * se * 100,
        "ppl": math.exp(2.3),
        "int8_scale": 0.5 / 127,
        "variance_ci": 1.96 * 30 / math.sqrt(5),
        "prefix_tflop": 2 * m["params_billion"] * 1e9 * 500 / 1e12,
        "cost_per_m": calc.cost_per_million_tokens(1.0, 3000),
        "j_per_token": calc.energy_per_token(b["tbp_w"], 3000),
        "tp_options": [t for t in range(1, m["q_heads"] + 1) if m["q_heads"] % t == 0],
    }


def _plain(row: dict, field: str) -> str:
    v = row[field]
    if field == "level":
        lv = load()["levels"][v]
        return f"{lv['icon']} {lv['name']}"
    if field == "depends_on":
        names = concept_names()
        return ", ".join(f"{d} {names[d]}" for d in v) or "none"
    if field == "links":
        srcs = sources()
        return " · ".join(f"{srcs[x]['title']} <{srcs[x]['url']}>" for x in v) or "—"
    if field == "step":
        return f"{row['icon']} {row['id']} · {v}"
    if field == "concept":
        return f"{row['id']} · {v}"
    return str(v)


def to_csv(rows: list[dict], columns) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id"] + [h for _, h in columns])
    for r in rows:
        w.writerow([r["id"]] + [_plain(r, f) for f, _ in columns])
    return buf.getvalue()


def to_markdown(concepts: list[dict], math_rows: list[dict]) -> str:
    def table(rows, columns):
        out = ["| " + " | ".join(h for _, h in columns) + " |", "|" + "---|" * len(columns)]
        for r in rows:
            out.append("| " + " | ".join(_plain(r, f).replace("|", "\\|") for f, _ in columns) + " |")
        return out
    lines = ["# Senior TPM Concepts & Inference Math", "",
             "Example: Qwen/Qwen2.5-0.5B-Instruct served with vLLM; device numbers use the Intel Arc Pro B70. "
             "Worked numbers are computed from public specs or stated assumptions, not measurements.", "",
             "## Table 1 · Prerequisite concepts", ""] + table(concepts, CONCEPT_COLUMNS) + [
             "", "## Table 2 · Inference math, statistics and formulas", ""] + table(math_rows, MATH_COLUMNS)
    return "\n".join(lines) + "\n"
