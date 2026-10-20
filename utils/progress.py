"""Learner progress: kept in Streamlit session state, exported/imported as JSON.

There is deliberately no server-side progress file: on shared hosting such a file
would mix different learners' progress, and hosted file systems may be wiped on restart.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

SCHEMA_VERSION = 1
MAX_IMPORT_BYTES = 512 * 1024


def default_progress() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "completed_lessons": [],
        "check_results": {},      # lesson_id -> {"correct": int, "total": int}
        "diagnostic": None,       # {"by_topic": {...}, "recommended_path": str, "correct": int, "total": int}
        "labs_done": [],
        "scenarios": {},          # scenario_id -> {"reasoning": pct, "completeness": pct, "score": pct}
        "cases": {},              # case_id -> {"score": pct, "correct": bool}
        "notes": "",
    }


def export_json(progress: dict) -> str:
    data = {**progress, "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "app": "Intel AI/ML Technical TPM Academy"}
    return json.dumps(data, indent=2, sort_keys=True)


def _clean_pct(value) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def import_json(text: str | bytes, known_lessons: set[str], known_labs: set[str],
                known_scenarios: set[str], known_cases: set[str]) -> tuple[dict, list[str]]:
    """Parse and sanitize an exported progress file. Unknown ids are dropped with a warning.

    Raises ValueError for files that are not valid progress exports.
    """
    if isinstance(text, bytes):
        if len(text) > MAX_IMPORT_BYTES:
            raise ValueError("file is too large to be a progress export")
        text = text.decode("utf-8")
    elif len(text.encode("utf-8")) > MAX_IMPORT_BYTES:
        raise ValueError("file is too large to be a progress export")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("not a progress export from this app (schema_version missing or different)")

    warnings: list[str] = []
    out = default_progress()

    def known(items, allowed, label):
        kept = []
        for item in items if isinstance(items, list) else []:
            if isinstance(item, str) and item in allowed and item not in kept:
                kept.append(item)
            else:
                warnings.append(f"ignored unknown {label}: {item!r}")
        return kept

    out["completed_lessons"] = known(raw.get("completed_lessons", []), known_lessons, "lesson")
    out["labs_done"] = known(raw.get("labs_done", []), known_labs, "lab")

    for lid, res in (raw.get("check_results") or {}).items():
        if lid in known_lessons and isinstance(res, dict):
            total = int(res.get("total", 0) or 0)
            correct = max(0, min(total, int(res.get("correct", 0) or 0)))
            out["check_results"][lid] = {"correct": correct, "total": total}
        else:
            warnings.append(f"ignored check result for {lid!r}")

    for sid, res in (raw.get("scenarios") or {}).items():
        if sid in known_scenarios and isinstance(res, dict):
            out["scenarios"][sid] = {k: _clean_pct(res.get(k)) for k in ("reasoning", "completeness", "score")}
        else:
            warnings.append(f"ignored scenario {sid!r}")

    for cid, res in (raw.get("cases") or {}).items():
        if cid in known_cases and isinstance(res, dict):
            out["cases"][cid] = {"score": _clean_pct(res.get("score")), "correct": bool(res.get("correct"))}
        else:
            warnings.append(f"ignored case {cid!r}")

    diag = raw.get("diagnostic")
    if isinstance(diag, dict) and diag.get("recommended_path") in ("beginner", "working", "senior"):
        by_topic = {}
        for topic, val in (diag.get("by_topic") or {}).items():
            if isinstance(val, dict):
                by_topic[str(topic)] = {"correct": int(val.get("correct", 0) or 0), "total": int(val.get("total", 0) or 0)}
        out["diagnostic"] = {
            "by_topic": by_topic,
            "recommended_path": diag["recommended_path"],
            "correct": int(diag.get("correct", 0) or 0),
            "total": int(diag.get("total", 0) or 0),
        }
    out["notes"] = str(raw.get("notes", ""))[:20000]
    return out, warnings


def completion_pct(progress: dict, lesson_ids: list[str]) -> int:
    if not lesson_ids:
        return 0
    done = sum(1 for lid in lesson_ids if lid in progress["completed_lessons"])
    return round(100 * done / len(lesson_ids))
