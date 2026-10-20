"""Content schema and coverage tests."""
import re

from utils.content import EXERCISE_LABELS, all_checks, load_all, validate

C = load_all()


def test_schema_is_valid():
    assert validate() == []


def test_minimum_counts():
    assert len(C["lessons"]) >= 20
    assert len(all_checks()) >= 40
    assert len(C["diagnostic"]["questions"]) == 15
    assert len(C["glossary"]) >= 60
    assert len(C["scenarios"]) >= 5
    laptop = [lab for lab in C["labs"] if lab["label"] == "Runs on a normal laptop"]
    assert len(laptop) >= 8
    assert len(C["templates"]) == 9


def test_every_track_has_lessons():
    for track in C["tracks"]:
        assert any(lesson["path"] == track for lesson in C["lessons"]), track


def test_diagnostic_covers_each_topic_three_times():
    topics = [q["topic"] for q in C["diagnostic"]["questions"]]
    for topic in C["diagnostic"]["topics"]:
        assert topics.count(topic) == 3


def test_required_troubleshooting_cases_present():
    titles = " ".join(case["title"].lower() for case in C["cases"])
    for needle in ["driver/runtime mismatch", "not detected", "out-of-memory", "slow inference",
                   "quantization", "network bottleneck", "unsupported model/operator"]:
        assert needle in titles, needle


def test_lesson_has_all_eleven_parts():
    for lesson in C["lessons"]:
        assert lesson["explanation"].strip() and lesson["why_it_matters"].strip()
        assert lesson["customer_example"].strip() and lesson["visual"]
        assert lesson["exercise"]["label"] in EXERCISE_LABELS
        assert lesson["expected_output"].strip() and lesson["interpretation"].strip()
        assert len(lesson["misconceptions"]) >= 2 and len(lesson["senior_questions"]) >= 2
        assert len(lesson["checks"]) >= 1 and lesson["sources"]


def test_source_metadata_complete_and_honest():
    for sid, src in C["sources"].items():
        assert src["url"].startswith("https://"), sid
        assert src["status"], sid
        if src["checked"] is not None:
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", src["checked"]), sid
            assert src["checked"] <= "2026-10-20", f"{sid}: check date in the future"
        else:
            assert src["status"] == "unverified", sid


def test_third_party_sources_are_not_marked_verified_product_facts():
    for sid, src in C["sources"].items():
        if src["kind"] == "third-party":
            assert src["status"] == "unverified", sid


def test_product_specific_lessons_cite_official_or_upstream():
    for lesson in C["lessons"]:
        if lesson["product_specific"]:
            kinds = {C["sources"][s]["kind"] for s in lesson["sources"]}
            assert kinds & {"official", "upstream"}, lesson["id"]


def test_codenames_flagged():
    crescent = next(p for p in C["products"] if p["id"] == "crescent-island")
    assert crescent["status"] == "announced"
    assert "verif" in crescent["tpm_notes"].lower() or "don't quote" in crescent["tpm_notes"].lower()


def test_hardware_exercises_name_hardware():
    for lesson in C["lessons"]:
        if lesson["exercise"]["label"] == "Requires specified Intel hardware":
            assert lesson["exercise"].get("hardware")
    for lab in C["labs"]:
        if lab["label"] == "Requires specified Intel hardware":
            assert lab.get("hardware") and lab.get("runner") is None


def test_simulated_outputs_are_labeled():
    # Hardware-only exercises must not present their expected output as a real run.
    for lesson in C["lessons"]:
        if lesson["exercise"]["label"] == "Requires specified Intel hardware":
            assert "illustrative" in lesson["expected_output"].lower()
    for case in C["cases"]:
        assert case["actions"], case["id"]


def test_answer_positions_are_varied():
    answers = [q["answer"] for q in all_checks()]
    assert len(set(answers)) >= 3


def test_templates_are_marked_fictional():
    for key, tpl in C["templates"].items():
        assert "fictional" in tpl["body"].lower(), key
