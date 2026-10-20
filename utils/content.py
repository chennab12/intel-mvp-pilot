"""Load and validate the app's content files (YAML and Markdown under content/).

Everything here is plain Python so tests can run without Streamlit.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = ROOT / "content"

EXERCISE_LABELS = ("Runs on a normal laptop", "Simulation only", "Requires specified Intel hardware")
SOURCE_STATUSES = ("available", "announced", "historical", "deprecated", "unverified", "concept")
PRODUCT_STATUSES = ("available", "announced", "historical", "deprecated", "unverified")
SOURCE_KINDS = ("official", "upstream", "paper", "third-party", "original")
VISUAL_TYPES = ("table", "dot", "bars")
LESSON_FIELDS = (
    "id", "title", "level", "minutes", "product_specific", "summary", "explanation", "why_it_matters",
    "customer_example", "visual", "exercise", "expected_output", "interpretation", "misconceptions",
    "senior_questions", "checks", "sources",
)


def _read_yaml(name: str):
    with open(CONTENT_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _date_str(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()[:10]
    return str(value)


@lru_cache(maxsize=1)
def load_all() -> dict:
    """Read every content file once and return one dictionary."""
    tracks_doc = _read_yaml("paths.yaml")
    lessons = []
    for file in sorted((CONTENT_DIR / "lessons").glob("*.yaml")):
        doc = _read_yaml(f"lessons/{file.name}")
        for lesson in doc["lessons"]:
            lessons.append({**lesson, "path": doc["path"]})

    sources = _read_yaml("sources.yaml")["sources"]
    for src in sources.values():
        src["checked"] = _date_str(src.get("checked"))
        src["review_by"] = _date_str(src.get("review_by"))

    templates = {}
    for file in sorted((CONTENT_DIR / "templates").glob("*.md")):
        text = file.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        templates[file.stem] = {"title": title, "body": text, "filename": file.name}

    return {
        "tracks": tracks_doc["tracks"],
        "learning_paths": tracks_doc["learning_paths"],
        "recommended_first": tracks_doc["recommended_first"],
        "lessons": lessons,
        "lessons_by_id": {lesson["id"]: lesson for lesson in lessons},
        "sources": sources,
        "diagnostic": _read_yaml("diagnostic.yaml"),
        "glossary": _read_yaml("glossary.yaml")["terms"],
        "labs": _read_yaml("labs.yaml")["labs"],
        "scenarios": _read_yaml("scenarios.yaml")["scenarios"],
        "cases": _read_yaml("troubleshooting.yaml")["cases"],
        "products": _read_yaml("products.yaml")["products"],
        "stack": _read_yaml("stack.yaml")["paths"],
        "templates": templates,
    }


def source_label(src: dict) -> str:
    """Short human label for a source's verification state."""
    if src.get("checked") and src.get("status") != "unverified":
        return f"Checked {src['checked']}"
    if src.get("checked"):
        return f"Opened {src['checked']} - claim needs verification"
    return "Needs verification"


def all_checks(content: dict | None = None) -> list[dict]:
    """Every knowledge-check question, tagged with its lesson."""
    content = content or load_all()
    return [
        {**check, "lesson_id": lesson["id"], "lesson_title": lesson["title"], "path": lesson["path"]}
        for lesson in content["lessons"]
        for check in lesson["checks"]
    ]


def _check_question(item: dict, where: str, errors: list[str]) -> None:
    for key in ("question", "options", "answer", "explanation"):
        if key not in item:
            errors.append(f"{where}: missing {key}")
            return
    if not isinstance(item["options"], list) or len(item["options"]) < 2:
        errors.append(f"{where}: needs at least two options")
    elif not (isinstance(item["answer"], int) and 0 <= item["answer"] < len(item["options"])):
        errors.append(f"{where}: answer index out of range")
    if not str(item["explanation"]).strip():
        errors.append(f"{where}: empty explanation")


def validate(content: dict | None = None) -> list[str]:
    """Return a list of schema problems. An empty list means the content is valid."""
    content = content or load_all()
    errors: list[str] = []
    sources = content["sources"]
    lab_ids = {lab["id"] for lab in content["labs"]}

    for sid, src in sources.items():
        for key in ("title", "publisher", "url", "kind", "status", "product", "facts", "review_by"):
            if not src.get(key):
                errors.append(f"source {sid}: missing {key}")
        if "checked" not in src:
            errors.append(f"source {sid}: missing checked (use null if never checked)")
        if src.get("status") not in SOURCE_STATUSES:
            errors.append(f"source {sid}: bad status {src.get('status')}")
        if src.get("kind") not in SOURCE_KINDS:
            errors.append(f"source {sid}: bad kind {src.get('kind')}")
        if not str(src.get("url", "")).startswith("https://"):
            errors.append(f"source {sid}: url must be https")
        if src.get("checked") is None and src.get("status") != "unverified":
            errors.append(f"source {sid}: never checked, so status must be unverified")

    seen = set()
    for lesson in content["lessons"]:
        lid = lesson.get("id", "?")
        if lid in seen:
            errors.append(f"lesson {lid}: duplicate id")
        seen.add(lid)
        for key in LESSON_FIELDS:
            if key not in lesson or lesson[key] in (None, "", []):
                errors.append(f"lesson {lid}: missing {key}")
        if lesson.get("path") not in content["tracks"]:
            errors.append(f"lesson {lid}: unknown path {lesson.get('path')}")
        visual = lesson.get("visual") or {}
        if visual.get("type") not in VISUAL_TYPES:
            errors.append(f"lesson {lid}: bad visual type")
        elif visual["type"] == "table":
            for row in visual.get("rows", []):
                if len(row) != len(visual.get("columns", [])):
                    errors.append(f"lesson {lid}: table row width mismatch {row}")
        elif visual["type"] == "bars" and len(visual.get("x", [])) != len(visual.get("y", [])):
            errors.append(f"lesson {lid}: bars x/y length mismatch")
        elif visual["type"] == "dot" and "digraph" not in visual.get("source", ""):
            errors.append(f"lesson {lid}: dot visual needs a digraph")
        exercise = lesson.get("exercise") or {}
        if exercise.get("label") not in EXERCISE_LABELS:
            errors.append(f"lesson {lid}: bad exercise label {exercise.get('label')}")
        if exercise.get("label") == "Requires specified Intel hardware" and not exercise.get("hardware"):
            errors.append(f"lesson {lid}: hardware exercise must name the hardware")
        if exercise.get("lab") and exercise["lab"] not in lab_ids:
            errors.append(f"lesson {lid}: unknown lab {exercise['lab']}")
        for i, check in enumerate(lesson.get("checks", [])):
            _check_question(check, f"lesson {lid} check {i}", errors)
        for sid in lesson.get("sources", []):
            if sid not in sources:
                errors.append(f"lesson {lid}: unknown source {sid}")
        if lesson.get("product_specific"):
            product_sources = [sources[s] for s in lesson.get("sources", []) if s in sources]
            if not any(s["kind"] in ("official", "upstream") for s in product_sources):
                errors.append(f"lesson {lid}: product-specific lesson needs an official or upstream source")

    lessons_by_id = content["lessons_by_id"]
    for path in content["learning_paths"]:
        for lid in path["lessons"]:
            if lid not in lessons_by_id:
                errors.append(f"learning path {path['id']}: unknown lesson {lid}")
    for lid in content["recommended_first"]:
        if lid not in lessons_by_id:
            errors.append(f"recommended_first: unknown lesson {lid}")

    diag = content["diagnostic"]
    for i, q in enumerate(diag["questions"]):
        _check_question(q, f"diagnostic {i}", errors)
        if q.get("topic") not in diag["topics"]:
            errors.append(f"diagnostic {i}: unknown topic")
        if q.get("lesson") not in lessons_by_id:
            errors.append(f"diagnostic {i}: unknown lesson {q.get('lesson')}")

    for term in content["glossary"]:
        if not term.get("term") or not term.get("definition") or not term.get("category"):
            errors.append(f"glossary: incomplete term {term}")
        if term.get("source") and term["source"] not in sources:
            errors.append(f"glossary {term['term']}: unknown source {term['source']}")

    for lab in content["labs"]:
        if lab.get("label") not in EXERCISE_LABELS:
            errors.append(f"lab {lab['id']}: bad label")
        if lab.get("label") == "Requires specified Intel hardware" and not lab.get("hardware"):
            errors.append(f"lab {lab['id']}: hardware lab must name the hardware")
        if lab.get("lesson") not in lessons_by_id:
            errors.append(f"lab {lab['id']}: unknown lesson")

    for sc in content["scenarios"]:
        nodes, endings = sc["nodes"], sc["endings"]
        if sc["start"] not in nodes:
            errors.append(f"scenario {sc['id']}: start node missing")
        for nid, node in nodes.items():
            for choice in node["choices"]:
                if choice["next"] not in nodes and choice["next"] not in endings:
                    errors.append(f"scenario {sc['id']} {nid}: unknown next {choice['next']}")
                for ev in choice.get("evidence", []):
                    if ev not in sc["required_evidence"]:
                        errors.append(f"scenario {sc['id']} {nid}: unknown evidence {ev}")

    for case in content["cases"]:
        if sum(1 for d in case["diagnoses"] if d["correct"]) != 1:
            errors.append(f"case {case['id']}: needs exactly one correct diagnosis")
        if not any(a.get("key") for a in case["actions"]):
            errors.append(f"case {case['id']}: needs key evidence")
        for sid in case.get("sources", []):
            if sid not in sources:
                errors.append(f"case {case['id']}: unknown source {sid}")

    for product in content["products"]:
        if product.get("status") not in PRODUCT_STATUSES:
            errors.append(f"product {product['id']}: bad status")
        for sid in product.get("sources", []):
            if sid not in sources:
                errors.append(f"product {product['id']}: unknown source {sid}")
        if not product.get("sources"):
            errors.append(f"product {product['id']}: no sources")

    for path in content["stack"]:
        for layer in path["layers"]:
            if layer.get("source") and layer["source"] not in sources:
                errors.append(f"stack {path['id']}: unknown source {layer['source']}")

    return errors
