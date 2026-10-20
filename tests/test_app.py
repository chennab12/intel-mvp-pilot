"""End-to-end UI tests with Streamlit's AppTest (no browser needed)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from utils.content import load_all

SECTIONS = ["Start Here", "Learning Paths", "Intel Product Explorer", "Software Stack Explorer", "Practical Labs",
            "Customer POC Simulator", "Performance Playground", "Troubleshooting Clinic", "Senior TPM Toolkit",
            "My Progress", "Glossary and Sources", "Model Functional Testing", "Model Performance Testing",
            "Agentic AI Inference Lab", "Golden Master Cheatsheet",
            "Senior TPM Concepts & Inference Math"]
C = load_all()


def app():
    return AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=60).run()


def nav(at, section):
    at.session_state["nav"] = section
    return at.run()


def button(at, label_start):
    return next(b for b in at.button if b.label.startswith(label_start))


def test_startup():
    at = app()
    assert not at.exception
    assert at.title[0].value.startswith("🚀 Start Here")


@pytest.mark.parametrize("section", SECTIONS)
def test_every_section_renders(section):
    at = nav(app(), section)
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.markdown) > 0


def test_global_search_opens_lesson():
    at = app()
    at.sidebar.text_input[0].set_value("KV cache").run()
    assert any("Search results" in m.value for m in at.markdown)
    open_buttons = [b for b in at.button if b.label == "Open"]
    assert open_buttons
    open_buttons[0].click().run()
    assert not at.exception
    assert at.session_state["global_search"] == ""


def test_diagnostic_scores_by_topic():
    at = app()
    questions = C["diagnostic"]["questions"]
    for i, q in enumerate(questions):
        at.radio(key=f"diag_{i}").set_value(q["answer"])
    next(b for b in at.button if b.label == "Submit diagnostic").click().run()
    assert not at.exception
    diag = at.session_state["progress"]["diagnostic"]
    assert diag["correct"] == 15 and diag["recommended_path"] == "senior"
    assert set(diag["by_topic"]) == set(C["diagnostic"]["topics"])


def test_lesson_knowledge_check_and_completion():
    at = nav(app(), "Learning Paths")
    lid = at.session_state["lesson_pick"]
    lesson = C["lessons_by_id"][lid]
    for i, q in enumerate(lesson["checks"]):
        at.radio(key=f"kc_{lid}_{i}").set_value(q["answer"])
    button(at, "Check my answers").click().run()
    res = at.session_state["progress"]["check_results"][lid]
    assert res == {"correct": len(lesson["checks"]), "total": len(lesson["checks"])}
    assert any("correct." in s.value for s in at.success)
    button(at, "✅ Mark lesson complete").click().run()
    assert lid in at.session_state["progress"]["completed_lessons"]


def test_wrong_answer_shows_explanation():
    at = nav(app(), "Learning Paths")
    lid = at.session_state["lesson_pick"]
    q = C["lessons_by_id"][lid]["checks"][0]
    at.radio(key=f"kc_{lid}_0").set_value((q["answer"] + 1) % len(q["options"]))
    button(at, "Check my answers").click().run()
    assert any(q["explanation"] in e.value for e in at.error)


def test_every_lesson_renders():
    at = nav(app(), "Learning Paths")
    at.selectbox(key="lp_path").set_value("All lessons (by track)").run()
    for lesson in C["lessons"]:
        at.selectbox(key="lesson_pick").set_value(lesson["id"]).run()
        assert not at.exception, (lesson["id"], [e.value for e in at.exception])
        assert at.header[0].value == lesson["title"]


def test_practice_quiz():
    at = nav(app(), "Learning Paths")
    at.radio(key="lp_view").set_value("Practice quiz").run()
    next(b for b in at.button if b.label == "Submit quiz").click().run()
    assert not at.exception
    assert any("Score:" in s.value for s in at.success)


@pytest.mark.parametrize("tool", ["Model weight memory", "KV-cache memory", "Total memory budget",
                                  "Throughput vs latency (simulated)", "Benchmark comparison", "Cost per million tokens"])
def test_performance_tools(tool):
    at = nav(app(), "Performance Playground")
    at.radio(key="perf_tool").set_value(tool).run()
    assert not at.exception, [e.value for e in at.exception]


def test_weight_calculator_known_value():
    at = nav(app(), "Performance Playground")
    values = {m.label: m.value for m in at.metric}
    assert values["Weights (GB, decimal)"] == "14.00"
    assert values["Weights (GiB, binary)"] == "13.04"


def test_kv_calculator_known_value():
    at = nav(app(), "Performance Playground")
    at.radio(key="perf_tool").set_value("KV-cache memory").run()
    values = {m.label: m.value for m in at.metric}
    assert values["Per token"] == "128.0 KiB"
    assert values["Total (GiB)"] == "16.00"


def test_cost_calculator_known_value():
    at = nav(app(), "Performance Playground")
    at.radio(key="perf_tool").set_value("Cost per million tokens").run()
    values = {m.label: m.value for m in at.metric}
    assert values["Cost per 1M tokens"] == "$2.78"


def test_benchmark_comparison_flags_default_rows():
    at = nav(app(), "Performance Playground")
    at.radio(key="perf_tool").set_value("Benchmark comparison").run()
    assert any("Not a fair comparison" in e.value for e in at.error)


def test_run_every_runnable_lab():
    at = nav(app(), "Practical Labs")
    for lab in C["labs"]:
        if not lab.get("runner"):
            continue
        at.selectbox(key="lab_pick").set_value(lab["id"]).run()
        at.button(key=f"run_{lab['id']}").click().run()
        assert not at.exception, (lab["id"], [e.value for e in at.exception])
        assert at.session_state[f"lab_result_{lab['id']}"]["kind"] in ("measured", "computed", "simulated")


def test_poc_simulator_play_through():
    at = nav(app(), "Customer POC Simulator")
    sc = C["scenarios"][0]
    for _ in range(10):
        run = at.session_state[f"poc_run_{sc['id']}"]
        if run["done"]:
            break
        node = sc["nodes"][run["node"]]
        best = max(range(len(node["choices"])), key=lambda i: node["choices"][i].get("points", 0))
        at.button(key=f"{sc['id']}_{run['node']}_{best}_{len(run['history'])}").click().run()
    assert at.session_state[f"poc_run_{sc['id']}"]["done"]
    assert at.session_state["progress"]["scenarios"][sc["id"]]["score"] == 100


def test_clinic_case_flow():
    at = nav(app(), "Troubleshooting Clinic")
    case = C["cases"][0]
    for action in case["actions"]:
        if action.get("key"):
            at.button(key=f"{case['id']}_{action['id']}").click().run()
    correct = next(i for i, d in enumerate(case["diagnoses"]) if d["correct"])
    at.radio(key=f"{case['id']}_diag_radio").set_value(correct).run()
    at.button(key=f"{case['id']}_submit").click().run()
    assert not at.exception
    assert at.session_state["progress"]["cases"][case["id"]] == {"score": 100, "correct": True}


def test_toolkit_templates_render():
    at = nav(app(), "Senior TPM Toolkit")
    for key in C["templates"]:
        at.selectbox(key="tk_pick").set_value(key).run()
        assert not at.exception
        assert "fictional" in at.text_area(key=f"tk_text_{key}").value.lower()


def test_glossary_search_and_registry():
    at = nav(app(), "Glossary and Sources")
    at.text_input(key="gl_q").set_value("RoCE").run()
    assert not at.exception
    at.radio(key="gl_view").set_value("Source registry").run()
    assert any("Awaiting verification" in w.value for w in at.warning)
    at.radio(key="gl_view").set_value("Verification workflow").run()
    assert any("schema check" in s.value for s in at.success)


def test_product_explorer_filter():
    at = nav(app(), "Intel Product Explorer")
    at.multiselect(key="pe_status").set_value(["announced"]).run()
    assert not at.exception
    assert any("Crescent Island" in e.label for e in at.expander)


@pytest.mark.parametrize("fault", ["Label order swapped in the deployed config", "Tokenizer mismatch (deployed tokenizer drops words)"])
def test_functional_page_shows_failure_for_injected_bug(fault):
    at = nav(app(), "Model Functional Testing")
    at.selectbox(key="mf_fault").set_value(fault).run()
    assert not at.exception
    assert any("First failure" in e.value for e in at.error)


def test_performance_page_load_test_and_report():
    at = nav(app(), "Model Performance Testing")
    assert not at.exception
    assert len(at.session_state["mp_last_load"]) == 2
    assert any(m.label == "Goodput" for m in at.metric)


def test_agent_page_blocks_destructive_request():
    at = nav(app(), "Agentic AI Inference Lab")
    at.text_input(key="ag_custom").set_value("please rm -rf /tmp on poc-node-01").run()
    assert not at.exception
    assert any("blocked" in e.value for e in at.error)


def test_cheatsheet_filters_and_card_view():
    at = nav(app(), "Golden Master Cheatsheet")
    assert any("gm-wrap" in m.value for m in at.markdown)
    at.text_input(key="gm_query").set_value("TTFT").run()
    at.radio(key="gm_view").set_value("Step cards").run()
    assert not at.exception
    assert any("TTFT" in e.label for e in at.expander)


def test_concepts_math_two_tables_and_filters():
    at = nav(app(), "Senior TPM Concepts & Inference Math")
    assert sum("class='cmt'" in m.value for m in at.markdown) == 2
    at.text_input(key="cm_q2").set_value("percentile").run()
    assert not at.exception
    tables = [m.value for m in at.markdown if "class='cmt'" in m.value]
    assert "M18" in tables[1] and "M03" not in tables[1]
