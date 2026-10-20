"""Branching POC scenario engine and Troubleshooting Clinic scoring (pure functions)."""
from __future__ import annotations

from functools import lru_cache


def new_run(scenario: dict) -> dict:
    return {"node": scenario["start"], "history": [], "evidence": [], "points": 0, "done": False}


def choose(scenario: dict, run: dict, choice_index: int) -> dict:
    """Apply a choice and return a new run state."""
    if run["done"]:
        return run
    node = scenario["nodes"][run["node"]]
    choice = node["choices"][choice_index]
    evidence = list(run["evidence"])
    for ev in choice.get("evidence", []):
        if ev not in evidence:
            evidence.append(ev)
    nxt = choice["next"]
    return {
        "node": nxt,
        "history": run["history"] + [{"node": run["node"], "choice": choice_index, "label": choice["label"],
                                      "feedback": choice["feedback"], "points": choice.get("points", 0)}],
        "evidence": evidence,
        "points": run["points"] + choice.get("points", 0),
        "done": nxt in scenario["endings"],
    }


def max_points(scenario: dict) -> int:
    """Best total points achievable on any path from the start."""
    nodes, endings = scenario["nodes"], scenario["endings"]

    @lru_cache(maxsize=None)
    def best(node_id: str, depth: int) -> int:
        if node_id in endings or depth > 50:
            return 0
        return max(c.get("points", 0) + best(c["next"], depth + 1) for c in nodes[node_id]["choices"])

    return best(scenario["start"], 0)


def score_run(scenario: dict, run: dict) -> dict:
    required = list(scenario["required_evidence"])
    collected = [e for e in required if e in run["evidence"]]
    best = max_points(scenario) or 1
    reasoning = round(100 * min(run["points"], best) / best)
    completeness = round(100 * len(collected) / len(required)) if required else 100
    return {
        "reasoning": reasoning,
        "completeness": completeness,
        "score": round((reasoning + completeness) / 2),
        "missing": [e for e in required if e not in collected],
    }


def score_case(case: dict, gathered: list[str], diagnosis_index: int | None) -> dict:
    """Troubleshooting score: 50% key evidence gathered, 40% correct diagnosis, 10% avoiding premature fixes.

    A correct diagnosis made with less than half of the key evidence earns only half the diagnosis credit,
    because it couldn't be defended to a customer or engineering team.
    """
    actions = {a["id"]: a for a in case["actions"]}
    key = [a["id"] for a in case["actions"] if a.get("key")]
    premature = [a for a in gathered if actions.get(a, {}).get("kind") == "premature"]
    key_found = [a for a in key if a in gathered]
    evidence_ratio = len(key_found) / len(key) if key else 1.0
    correct = diagnosis_index is not None and case["diagnoses"][diagnosis_index]["correct"]
    diag_credit = 0.0
    if correct:
        diag_credit = 1.0 if evidence_ratio >= 0.5 else 0.5
    score = 50 * evidence_ratio + 40 * diag_credit + (10 if not premature else 0)
    return {
        "score": round(score),
        "correct": bool(correct),
        "key_found": key_found,
        "key_missing": [a for a in key if a not in gathered],
        "premature": premature,
    }
