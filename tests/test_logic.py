"""Progress export/import, POC simulator, clinic scoring, search and lab runners."""
import json

import pytest

from utils import labs
from utils.content import load_all
from utils.poc import choose, max_points, new_run, score_case, score_run
from utils.progress import default_progress, export_json, import_json
from utils.search import search

C = load_all()
LESSONS = {lesson["id"] for lesson in C["lessons"]}
LABS = {lab["id"] for lab in C["labs"]}
SCEN = {s["id"] for s in C["scenarios"]}
CASES = {x["id"] for x in C["cases"]}


def _import(text):
    return import_json(text, LESSONS, LABS, SCEN, CASES)


def test_progress_round_trip():
    p = default_progress()
    p["completed_lessons"] = ["software-layers", "prefill-decode-kv"]
    p["check_results"]["software-layers"] = {"correct": 2, "total": 2}
    p["labs_done"] = ["lab-warmup"]
    p["scenarios"]["northwind-sizing"] = {"reasoning": 80, "completeness": 100, "score": 90}
    p["cases"]["out-of-memory"] = {"score": 100, "correct": True}
    p["diagnostic"] = {"by_topic": {"foundations": {"correct": 3, "total": 3}}, "recommended_path": "working",
                       "correct": 11, "total": 15}
    p["notes"] = "remember NUMA"
    loaded, warnings = _import(export_json(p))
    assert warnings == []
    for key in ("completed_lessons", "check_results", "labs_done", "scenarios", "cases", "diagnostic", "notes"):
        assert loaded[key] == p[key], key


def test_import_drops_unknown_and_clamps():
    raw = default_progress()
    raw["completed_lessons"] = ["software-layers", "not-a-lesson", 5]
    raw["check_results"] = {"software-layers": {"correct": 9, "total": 2}, "nope": {"correct": 1, "total": 1}}
    raw["scenarios"] = {"northwind-sizing": {"reasoning": 500, "completeness": -3, "score": "x"}}
    loaded, warnings = _import(json.dumps(raw))
    assert loaded["completed_lessons"] == ["software-layers"]
    assert loaded["check_results"] == {"software-layers": {"correct": 2, "total": 2}}
    assert loaded["scenarios"]["northwind-sizing"] == {"reasoning": 100.0, "completeness": 0.0, "score": 0.0}
    assert len(warnings) >= 3


@pytest.mark.parametrize("bad", ["not json", "[]", json.dumps({"schema_version": 99}), json.dumps({"foo": 1})])
def test_import_rejects_bad_files(bad):
    with pytest.raises(ValueError):
        _import(bad)


def test_import_rejects_huge_file():
    with pytest.raises(ValueError):
        _import(b"x" * (600 * 1024))


def test_every_scenario_is_winnable_and_scored():
    for sc in C["scenarios"]:
        assert max_points(sc) > 0
        run = new_run(sc)
        steps = 0
        while not run["done"]:
            node = sc["nodes"][run["node"]]
            best = max(range(len(node["choices"])), key=lambda i: node["choices"][i].get("points", 0))
            run = choose(sc, run, best)
            steps += 1
            assert steps < 20
        res = score_run(sc, run)
        assert res["reasoning"] == 100, sc["id"]
        assert res["completeness"] == 100, (sc["id"], res["missing"])


def test_poor_choices_score_lower():
    sc = next(s for s in C["scenarios"] if s["id"] == "northwind-sizing")
    run = new_run(sc)
    while not run["done"]:
        node = sc["nodes"][run["node"]]
        worst = min(range(len(node["choices"])), key=lambda i: node["choices"][i].get("points", 0))
        run = choose(sc, run, worst)
    res = score_run(sc, run)
    assert res["score"] < 50 and res["missing"]


def test_case_scoring_rewards_evidence_first():
    case = next(x for x in C["cases"] if x["id"] == "accelerator-not-detected")
    correct = next(i for i, d in enumerate(case["diagnoses"]) if d["correct"])
    key = [a["id"] for a in case["actions"] if a.get("key")]
    full = score_case(case, key, correct)
    assert full["score"] == 100
    guess = score_case(case, [], correct)
    assert guess["score"] == 30  # half diagnosis credit + no premature fix bonus
    rushed = score_case(case, ["newcard"] + key, correct)
    assert rushed["score"] == 90
    wrong = score_case(case, key, (correct + 1) % len(case["diagnoses"]))
    assert wrong["score"] == 60 and not wrong["correct"]


def test_search_finds_across_content():
    kinds = {r["kind"] for r in search("KV cache")}
    assert {"Lesson", "Glossary"} <= kinds
    assert any(r["kind"] == "Product" for r in search("Gaudi"))
    assert search("") == []
    assert search("zzzz-no-such-term") == []


def test_all_lab_runners_run_with_defaults():
    for lab in C["labs"]:
        if lab.get("runner"):
            res = labs.RUNNERS[lab["runner"]]({})
            assert res["kind"] in ("measured", "computed", "simulated"), lab["id"]


def test_lesson_quantization_numbers_match_runner():
    assert round(labs.quantization_error_simple(), 3) == pytest.approx(0.009, abs=0.001)
    table = labs.quantization_error({})["table"]
    assert table[0]["bytes"] == 400000 and table[1]["bytes"] == 100000
    assert table[1]["mean_abs_error"] == pytest.approx(labs.quantization_error_simple(), abs=1e-5)
    outl = labs.quantization_error({"outliers": True})["table"][1]["mean_abs_error"]
    per_ch = labs.quantization_error({"outliers": True, "per_channel": True})["table"][1]["mean_abs_error"]
    assert per_ch < outl


def test_mre_lab():
    all_steps = [s for s, _ in labs.MRE_STEPS]
    assert labs.shrink_mre({"kept": all_steps})["fails"]
    minimal = labs.shrink_mre({"kept": sorted(labs.MRE_TRIGGER)})
    assert minimal["fails"] and minimal["minimal"]
    assert not labs.shrink_mre({"kept": ["generate"]})["fails"]


def test_lab_inputs_are_capped():
    res = labs.latency_percentiles({"runs": 10_000_000})
    assert res["metrics"]["count"] == 1000
