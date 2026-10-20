"""Simple in-memory search across lessons, glossary, products, labs, cases and sources."""
from __future__ import annotations

import re
from functools import lru_cache

from utils.content import load_all

SECTION_FOR_KIND = {
    "Lesson": "Learning Paths",
    "Glossary": "Glossary and Sources",
    "Source": "Glossary and Sources",
    "Product": "Intel Product Explorer",
    "Lab": "Practical Labs",
    "Troubleshooting case": "Troubleshooting Clinic",
    "Scenario": "Customer POC Simulator",
    "Stack": "Software Stack Explorer",
}


def _text(*parts) -> str:
    out = []
    for p in parts:
        if isinstance(p, (list, tuple)):
            out.append(" ".join(str(x) for x in p))
        elif isinstance(p, dict):
            out.append(" ".join(str(v) for v in p.values()))
        elif p:
            out.append(str(p))
    return " ".join(out)


@lru_cache(maxsize=1)
def build_index() -> list[dict]:
    c = load_all()
    items = []
    for lesson in c["lessons"]:
        items.append({"kind": "Lesson", "id": lesson["id"], "title": lesson["title"], "summary": lesson["summary"],
                      "text": _text(lesson["title"], lesson["summary"], lesson["explanation"], lesson["customer_example"],
                                    lesson["misconceptions"], lesson["senior_questions"])})
    for term in c["glossary"]:
        items.append({"kind": "Glossary", "id": term["term"], "title": term["term"], "summary": term["definition"],
                      "text": _text(term["term"], term["definition"], term["category"])})
    for p in c["products"]:
        items.append({"kind": "Product", "id": p["id"], "title": p["name"], "summary": p["ai_role"],
                      "text": _text(p["name"], p["key_facts"], p["ai_role"], p["software"], p["tpm_notes"], p["category"])})
    for lab in c["labs"]:
        items.append({"kind": "Lab", "id": lab["id"], "title": lab["title"], "summary": lab["objective"],
                      "text": _text(lab["title"], lab["objective"], lab["steps"], lab["label"])})
    for case in c["cases"]:
        items.append({"kind": "Troubleshooting case", "id": case["id"], "title": case["title"], "summary": case["symptom"],
                      "text": _text(case["title"], case["symptom"], case["platform"], case["resolution"])})
    for sc in c["scenarios"]:
        items.append({"kind": "Scenario", "id": sc["id"], "title": sc["title"], "summary": sc["brief"].strip(),
                      "text": _text(sc["title"], sc["brief"], sc["skills"])})
    for path in c["stack"]:
        items.append({"kind": "Stack", "id": path["id"], "title": path["name"], "summary": f"{len(path['layers'])} layers",
                      "text": _text(path["name"], [_text(layer) for layer in path["layers"]])})
    for sid, src in c["sources"].items():
        items.append({"kind": "Source", "id": sid, "title": src["title"], "summary": src["product"],
                      "text": _text(src["title"], src["product"], src["facts"], src["publisher"])})
    for item in items:
        item["_title"] = item["title"].lower()
        item["_text"] = item["text"].lower()
    return items


def search(query: str, limit: int = 25) -> list[dict]:
    """Rank items by matches of every query word; title matches count more."""
    words = [w for w in re.findall(r"[\w./+-]+", query.lower()) if len(w) > 1]
    if not words:
        return []
    results = []
    for item in build_index():
        if not all(w in item["_text"] or w in item["_title"] for w in words):
            continue
        score = sum(5 * item["_title"].count(w) + item["_text"].count(w) for w in words)
        results.append((score, item))
    results.sort(key=lambda pair: (-pair[0], pair[1]["title"]))
    return [{k: v for k, v in item.items() if not k.startswith("_")} | {"section": SECTION_FOR_KIND[item["kind"]]}
            for _, item in results[:limit]]
