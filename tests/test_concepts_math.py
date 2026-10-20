"""Senior TPM Concepts & Inference Math: content rules and every quoted number."""
import pytest

from utils import concepts_math as cm

SECTIONS = ["Start Here", "Learning Paths", "Intel Product Explorer", "Software Stack Explorer", "Practical Labs",
            "Customer POC Simulator", "Performance Playground", "Troubleshooting Clinic", "Senior TPM Toolkit",
            "My Progress", "Glossary and Sources", "Model Functional Testing", "Model Performance Testing",
            "Agentic AI Inference Lab", "Golden Master Cheatsheet", "Senior TPM Concepts & Inference Math"]


def test_content_valid_and_prerequisites_ordered():
    assert cm.validate(SECTIONS) == []


def test_coverage():
    d = cm.load()
    assert len(d["concepts"]) >= 35 and len(d["math"]) >= 20
    assert {c["level"] for c in d["concepts"]} == {"foundation", "working", "senior"}
    stages = [m["stage"] for m in d["math"]]
    assert stages.index("⚡ Performance") > max(i for i, s in enumerate(stages) if s == "✅ Functional")


def _text(ids):
    rows = {r["id"]: r for r in cm.load()["concepts"] + cm.load()["math"]}
    return " ".join(str(rows[i].get("example", "")) + " " + str(rows[i].get("numbers", "")) for i in ids)


def test_worked_numbers_match_the_text():
    n = cm.worked_numbers()
    assert n["kv_bytes_per_token"] == 12288 and n["weights_gb"] == pytest.approx(0.98)
    assert round(n["kv_mb_4k"]) == 50 and round(n["kv_gb_32k"], 2) == 0.40
    assert n["kv_space_gb"] == pytest.approx(27.82)
    assert round(n["kv_tokens_m"], 2) == 2.26 and n["max_4k_requests"] == 552
    assert round(n["blocks_16"], -2) == 141_500 and round(n["blocks_64"]) == 35_375
    assert round(n["attention_mflop_4k"]) == 352 and round(n["attention_mflop_4k"] / 980 * 100) == 36
    assert round(n["prefill_tflop_1k"], 1) == 1.0 and round(n["prefill_ms_at_10tflops"]) == 100
    assert round(n["decode_ms_b1"], 2) == 1.63 and round(n["decode_tok_s_b1"]) == 613
    assert round(n["batch32_ms"], 2) == 2.27 and round(n["batch32_tok_s"]) == 14_071 and round(n["batch32_per_user"]) == 440
    assert n["e2e_s"] == pytest.approx(1.375) and n["little_l"] == pytest.approx(13.75)
    assert round(n["acc_se_pts"], 1) == 2.8 and round(n["acc_ci_pts"], 1) == 5.5
    assert round(n["ppl"]) == 10 and round(n["variance_ci"]) == 26
    assert n["int8_scale"] / 2 <= 0.002 and round(n["prefix_tflop"], 2) == 0.49
    assert round(n["cost_per_m"], 3) == 0.093 and round(n["j_per_token"], 3) == 0.077
    assert n["tp_options"] == [1, 2, 7, 14]
    text = _text(["M03", "M04", "M11", "M12", "M13", "M14", "M16", "M17", "M19", "M21", "M22", "M23", "M24", "C20"])
    for s in ("12,288 B", "≈ 50 MB", "27.82 GB", "≈ 2.26 M tokens", "≈ 552", "352 MFLOP", "36%", "≈ 100 ms",
              "≈ 1.63 ms", "≈ 613 tok/s", "≈ 2.27 ms", "≈ 14,071 tok/s", "≈ 440 tok/s", "1.375 s", "13.75",
              "±26", "0.49 TFLOP", "{1, 2, 7, 14}", "$0.093", "0.077 J"):
        assert s in text, s


def test_exports():
    d = cm.load()
    md = cm.to_markdown(d["concepts"], d["math"])
    assert "## Table 1" in md and "## Table 2" in md and md.count("\n| C") == len(d["concepts"])
    assert cm.to_csv(d["math"], cm.MATH_COLUMNS).count("\n") == len(d["math"]) + 1
