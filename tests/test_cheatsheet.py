"""Golden Master Cheatsheet content and computed numbers."""
import pytest

from utils import cheatsheet as cs


def test_content_is_valid():
    assert cs.validate() == []


def test_covers_functional_then_performance_end_to_end():
    steps = cs.load()["steps"]
    assert len(steps) >= 20
    order = [s["phase"] for s in steps]
    assert order.index("performance") > max(i for i, p in enumerate(order) if p == "functional")
    assert order[-1] == "report"


def test_model_facts_match_config():
    m = cs.load()["model"]
    assert (m["layers"], m["q_heads"], m["kv_heads"], m["hidden_size"]) == (24, 14, 2, 896)
    assert m["head_dim"] == m["hidden_size"] / m["q_heads"] == 64


def test_computed_numbers_match_the_text():
    k = cs.key_numbers()
    assert k["kv_bytes_per_token"] == 12288
    assert k["bf16_weights_gb"] == pytest.approx(0.98)
    assert k["kv_gb_per_full_context"] == pytest.approx(0.40, abs=0.005)
    assert round(k["b70_decode_ceiling_tok_s"]) == 620
    assert round(k["laptop_decode_ceiling_tok_s"]) == 61
    assert k["b70_full_context_sequences"] == 69
    assert k["b70_kv_tokens"] / 1e6 == pytest.approx(2.26, abs=0.01)
    text = " ".join(" ".join(s["metrics"]) for s in cs.load()["steps"])
    for snippet in ("12,288 B", "0.40 GB", "≈ 620 tok/s", "≈ 61 tok/s", "≈ 69 full-context", "0.98 GB", "$0.56"):
        assert snippet in text


def test_exports():
    steps = cs.filtered_steps()
    md = cs.to_markdown(steps)
    assert md.count("\n| ") >= len(steps) and "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct" in md
    csv_text = cs.to_csv(steps)
    assert csv_text.splitlines()[0].startswith("id,phase")
    assert len(cs.filtered_steps(["performance"])) == sum(s["phase"] == "performance" for s in steps)
    assert cs.filtered_steps(query="synchronize")
