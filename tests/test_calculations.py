"""Calculator units and known example values."""
import math

import pytest

from utils import calculators as calc


def test_weight_memory_known_values():
    w = calc.weight_memory_bytes(7, "BF16")
    assert w == 14e9
    assert round(w / calc.GIB, 2) == 13.04
    assert calc.weight_memory_bytes(70, "BF16") / calc.GB == pytest.approx(140)
    assert calc.weight_memory_bytes(13, "FP16") / calc.GB == pytest.approx(26)
    assert calc.weight_memory_bytes(7, "INT4") / calc.GB == pytest.approx(3.5)


def test_kv_cache_known_values():
    per_token = calc.kv_cache_bytes(32, 8, 128, 2, 1, 1)
    assert per_token == 131072  # 128 KiB
    total = calc.kv_cache_bytes(32, 8, 128, 2, 8192, 16)
    assert total == 16 * calc.GIB
    assert calc.kv_cache_bytes(40, 8, 128, 2, 1) == 160 * calc.KIB
    # Out-of-memory case in the clinic: 40 sequences x 16K tokens = 80 GiB
    assert calc.kv_cache_bytes(32, 8, 128, 2, 16384, 40) == 80 * calc.GIB


def test_memory_budget_lesson_example_does_not_fit():
    w = calc.weight_memory_bytes(8, "BF16")
    kv = calc.kv_cache_bytes(32, 8, 128, 2, 8192, 16)
    res = calc.memory_budget(w, kv, 10, 0, 32)
    assert res["weights_gb"] == pytest.approx(16)
    assert res["kv_gb"] == pytest.approx(17.18, abs=0.01)
    assert not res["fits"]
    assert res["devices_needed"] == 2


def test_memory_budget_fits_with_headroom():
    res = calc.memory_budget(10 * calc.GB, 5 * calc.GB, 10, 1, 32)
    assert res["total_gb"] == pytest.approx(17.5)
    assert res["headroom_gb"] == pytest.approx(14.5)
    assert res["fits"]


def test_max_sequences_consistent_with_budget():
    w = calc.weight_memory_bytes(8, "BF16")
    per_seq = calc.kv_cache_bytes(32, 8, 128, 2, 4096, 1)
    n = calc.max_sequences(32, w, 10, 1, per_seq)
    assert calc.memory_budget(w, per_seq * n, 10, 1, 32)["fits"]
    assert not calc.memory_budget(w, per_seq * (n + 1), 10, 1, 32)["fits"]


def test_tpot_examples():
    assert calc.tpot(5.0, 0.5, 101) == pytest.approx(0.045)
    assert calc.tpot(3.0, 1.0, 41) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        calc.tpot(1, 0.5, 1)


def test_percentiles_match_numpy_linear():
    values = list(range(1, 101))
    assert calc.percentile(values, 50) == pytest.approx(50.5)
    assert calc.percentile(values, 95) == pytest.approx(95.05)
    assert calc.percentile(values, 99) == pytest.approx(99.01)
    assert calc.percentile([7.0], 99) == 7.0


def test_cost_per_million_examples():
    assert calc.cost_per_million_tokens(10, 1000) == pytest.approx(2.7778, abs=1e-3)
    assert calc.cost_per_million_tokens(10, 1000, 50) == pytest.approx(5.5556, abs=1e-3)
    assert calc.cost_per_million_tokens(15, 2500) == pytest.approx(1.6667, abs=1e-3)
    assert calc.cost_per_million_tokens(20, 3000) == pytest.approx(1.85, abs=0.01)
    with pytest.raises(ValueError):
        calc.cost_per_million_tokens(10, 0)


def test_energy_per_token():
    assert calc.energy_per_token(400, 2000) == pytest.approx(0.2)


def test_scaling_speedup_matches_lesson():
    assert round(calc.scaling_speedup(8, 0.10), 2) == 4.71
    assert calc.scaling_speedup(1, 0.5) == 1.0
    assert calc.scaling_speedup(8, 0.0) == pytest.approx(8)


def test_simulated_serving_shape():
    rows = calc.simulate_serving([1, 8, 64, 128])
    agg = [r["aggregate_tokens_per_s"] for r in rows]
    per_user = [r["per_user_tokens_per_s"] for r in rows]
    assert agg[0] < agg[1] < agg[2]
    assert per_user[0] > per_user[1] > per_user[2]
    assert rows[3]["ttft_p50_ms"] > rows[2]["ttft_p50_ms"]  # queueing above max batch
    assert calc.simulate_serving([4], seed=1) == calc.simulate_serving([4], seed=1)


def test_fairness_flags_mismatches():
    a = {"run": "a", "model": "m", "precision": "BF16", "input_tokens": 2000, "output_tokens": 256, "concurrency": 32,
         "warmup_excluded": "yes", "accuracy_gate": "pass", "data_label": "measured", "software": "v1"}
    assert calc.fairness_issues([a, dict(a, run="b")]) == []
    issues = calc.fairness_issues([a, dict(a, run="b", concurrency=64, data_label="")])
    assert any("concurrency" in i for i in issues)
    assert any("no data label" in i for i in issues)


def test_critical_path_default_plan():
    from utils.labs import DEFAULT_PLAN
    path, total = calc.critical_path(DEFAULT_PLAN)
    assert path == ["Discovery", "Hardware ship", "Install + smoke", "Accuracy", "Benchmark", "Readout"]
    assert total == 36


def test_critical_path_errors():
    with pytest.raises(ValueError):
        calc.critical_path([{"task": "a", "days": 1, "depends_on": "b"}, {"task": "b", "days": 1, "depends_on": "a"}])
    with pytest.raises(ValueError):
        calc.critical_path([{"task": "a", "days": 1, "depends_on": "missing"}])


def test_requirement_check():
    vague = calc.requirement_check("The system should be fast.")
    assert not any(vague.values())
    good = calc.requirement_check("P95 TTFT ≤ 1.0 s at 40 concurrent users, 1,500-token inputs, BF16 on the agreed configuration")
    assert all(good.values()), good


def test_update_check():
    res = calc.update_check("At risk - soak test failed; need a fix in the RC by Nov 20.")
    assert res["short_enough"] and res["has_status"] and res["has_reason"] and res["has_ask"]


def test_rough_tokens():
    assert calc.rough_token_count("") == 0
    assert calc.rough_token_count("Hello, world!") == 4


def test_scoring_and_recommendation():
    qs = [{"answer": 0, "topic": "foundations"}, {"answer": 1, "topic": "foundations"}, {"answer": 2, "topic": "performance"}]
    s = calc.score_answers(qs, [0, None, 2])
    assert s["correct"] == 2 and s["total"] == 3 and s["results"] == [True, False, True]
    by = calc.diagnostic_by_topic(qs, [0, None, 2])
    assert by == {"foundations": {"correct": 1, "total": 2}, "performance": {"correct": 1, "total": 1}}
    assert calc.recommend_path({"foundations": {"correct": 1, "total": 3}}) == "beginner"
    strong = {t: {"correct": 3, "total": 3} for t in
              ("foundations", "intel-judgment", "customer-engineering", "performance", "senior-tpm")}
    assert calc.recommend_path(strong) == "senior"
    mid = dict(strong, performance={"correct": 1, "total": 3})
    assert calc.recommend_path(mid) == "working"


def test_negative_inputs_rejected():
    with pytest.raises(ValueError):
        calc.weight_memory_bytes(-1, "BF16")
    with pytest.raises(ValueError):
        calc.kv_cache_bytes(-1, 8, 128, 2, 10)
    assert math.isfinite(calc.weight_memory_bytes(0, "FP32"))
