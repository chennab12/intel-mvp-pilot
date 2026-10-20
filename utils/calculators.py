"""Pure calculation functions used by the Performance Playground, labs and scoring.

Every function here is arithmetic or a clearly labeled simulation. None of them
predicts the real performance of any product.
"""
from __future__ import annotations

import math
import random
import re

GB = 1e9           # decimal gigabyte
GIB = 1024 ** 3    # binary gibibyte
KIB = 1024

BYTES_PER_PARAM = {
    "FP32": 4.0,
    "BF16": 2.0,
    "FP16": 2.0,
    "FP8": 1.0,
    "INT8": 1.0,
    "INT4": 0.5,
}


# ---------- memory ----------

def weight_memory_bytes(params_billion: float, precision: str) -> float:
    """Approximate weight memory: parameters x bytes per parameter."""
    if params_billion < 0:
        raise ValueError("parameters must be >= 0")
    return params_billion * 1e9 * BYTES_PER_PARAM[precision]


def kv_cache_bytes(layers: int, kv_heads: int, head_dim: int, bytes_per_elem: float,
                   tokens: int, sequences: int = 1) -> float:
    """Simplified KV cache: 2 (K and V) x layers x KV heads x head dim x bytes x tokens x sequences."""
    for name, value in (("layers", layers), ("kv_heads", kv_heads), ("head_dim", head_dim),
                        ("tokens", tokens), ("sequences", sequences)):
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
    return 2 * layers * kv_heads * head_dim * bytes_per_elem * tokens * sequences


def memory_budget(weights_bytes: float, kv_bytes: float, overhead_pct: float,
                  fixed_overhead_gb: float, device_gb: float) -> dict:
    """Total = weights + KV + overhead_pct% of (weights + KV) + fixed overhead. Returns GB values."""
    variable = weights_bytes + kv_bytes
    overhead = variable * overhead_pct / 100 + fixed_overhead_gb * GB
    total = variable + overhead
    capacity = device_gb * GB
    return {
        "weights_gb": weights_bytes / GB,
        "kv_gb": kv_bytes / GB,
        "overhead_gb": overhead / GB,
        "total_gb": total / GB,
        "device_gb": device_gb,
        "headroom_gb": (capacity - total) / GB,
        "fits": total <= capacity,
        "devices_needed": max(1, math.ceil(total / capacity)) if capacity > 0 else None,
    }


def max_sequences(device_gb: float, weights_bytes: float, overhead_pct: float, fixed_overhead_gb: float,
                  kv_bytes_per_sequence: float) -> int:
    """How many full-length sequences fit after weights and overhead (same overhead model as memory_budget)."""
    if kv_bytes_per_sequence <= 0:
        return 0
    available = device_gb * GB - fixed_overhead_gb * GB
    per_seq = kv_bytes_per_sequence * (1 + overhead_pct / 100)
    room = available - weights_bytes * (1 + overhead_pct / 100)
    return max(0, int(room // per_seq))


# ---------- latency & throughput ----------

def tpot(e2e_s: float, ttft_s: float, output_tokens: int) -> float:
    """Time per output token (seconds), as defined by vLLM's benchmark: (E2E - TTFT) / (output tokens - 1)."""
    if output_tokens < 2:
        raise ValueError("need at least 2 output tokens")
    return (e2e_s - ttft_s) / (output_tokens - 1)


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (same method as numpy's default)."""
    if not values:
        raise ValueError("no values")
    data = sorted(values)
    pos = (len(data) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def latency_summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def cost_per_million_tokens(hourly_cost: float, tokens_per_second: float, utilization_pct: float = 100) -> float:
    """Illustrative cost per 1M tokens = hourly cost / (tokens/s x 3600 x utilization) x 1e6."""
    if tokens_per_second <= 0 or utilization_pct <= 0:
        raise ValueError("throughput and utilization must be > 0")
    effective_tps = tokens_per_second * utilization_pct / 100
    return hourly_cost / (effective_tps * 3600) * 1e6


def energy_per_token(avg_power_w: float, tokens_per_second: float) -> float:
    """Joules per token = watts / tokens per second."""
    if tokens_per_second <= 0:
        raise ValueError("tokens per second must be > 0")
    return avg_power_w / tokens_per_second


def scaling_speedup(n_devices: int, comm_fraction: float) -> float:
    """Toy model: compute splits evenly; a fixed share of single-device time becomes communication when n > 1."""
    if n_devices < 1:
        raise ValueError("need at least one device")
    comm = comm_fraction if n_devices > 1 else 0.0
    return 1 / ((1 - comm_fraction) / n_devices + comm) if n_devices > 1 else 1.0


def simulate_serving(concurrency_levels: list[int], *, weight_read_ms: float = 20.0, per_seq_ms: float = 0.6,
                     prefill_ms_per_1k: float = 40.0, input_tokens: int = 1000, output_tokens: int = 200,
                     max_batch: int = 64, jitter_pct: float = 8.0, seed: int = 7) -> list[dict]:
    """SIMULATED serving curve for teaching. Not a model of any real device.

    Decode step time = weight_read_ms + per_seq_ms x active batch (memory-bound toy model).
    Requests above max_batch wait in a queue, which adds to TTFT.
    """
    rng = random.Random(seed)
    rows = []
    for c in concurrency_levels:
        batch = min(c, max_batch)
        step_ms = weight_read_ms + per_seq_ms * batch
        prefill_ms = prefill_ms_per_1k * input_tokens / 1000 * (1 + batch / max_batch)
        waves = math.ceil(c / max_batch)
        queue_ms = (waves - 1) * (prefill_ms + step_ms * output_tokens) / 2
        ttft_ms = queue_ms + prefill_ms
        per_user_tps = 1000 / step_ms
        aggregate_tps = per_user_tps * batch
        samples = [ttft_ms * (1 + rng.gauss(0, jitter_pct / 100)) for _ in range(200)]
        samples = [max(1.0, s) for s in samples]
        rows.append({
            "concurrency": c,
            "decode_step_ms": round(step_ms, 2),
            "per_user_tokens_per_s": round(per_user_tps, 1),
            "aggregate_tokens_per_s": round(aggregate_tps, 1),
            "ttft_p50_ms": round(percentile(samples, 50), 1),
            "ttft_p95_ms": round(percentile(samples, 95), 1),
            "e2e_p50_s": round((percentile(samples, 50) + step_ms * (output_tokens - 1)) / 1000, 2),
        })
    return rows


# ---------- benchmark fairness ----------

FAIRNESS_KEYS = ("model", "precision", "input_tokens", "output_tokens", "concurrency", "warmup_excluded",
                 "accuracy_gate")


def fairness_issues(runs: list[dict], keys: tuple[str, ...] = FAIRNESS_KEYS) -> list[str]:
    """List settings that differ between runs being compared."""
    issues = []
    if len(runs) < 2:
        return issues
    for key in keys:
        values = {str(r.get(key, "")).strip().lower() for r in runs}
        if len(values) > 1:
            issues.append(f"{key} differs: " + ", ".join(sorted(str(r.get(key, "")) for r in runs)))
    for r in runs:
        if str(r.get("data_label", "")).strip().lower() not in ("measured", "vendor claim", "estimate", "simulated"):
            issues.append(f"run '{r.get('run', '?')}' has no data label (measured / vendor claim / estimate / simulated)")
        if not str(r.get("software", "")).strip():
            issues.append(f"run '{r.get('run', '?')}' has no software versions recorded")
    return issues


# ---------- planning ----------

def critical_path(tasks: list[dict]) -> tuple[list[str], float]:
    """Longest path through tasks [{'task', 'days', 'depends_on': 'a, b'}]. Raises on cycles or unknown deps."""
    names = [t["task"].strip() for t in tasks]
    by_name = {t["task"].strip(): t for t in tasks}
    deps = {}
    for t in tasks:
        raw = t.get("depends_on") or ""
        items = [d.strip() for d in str(raw).split(",") if d.strip()]
        for d in items:
            if d not in by_name:
                raise ValueError(f"unknown dependency '{d}' for task '{t['task']}'")
        deps[t["task"].strip()] = items
    finish, prev, state = {}, {}, {}

    def visit(name):
        if state.get(name) == "done":
            return finish[name]
        if state.get(name) == "visiting":
            raise ValueError(f"cycle detected at '{name}'")
        state[name] = "visiting"
        start, best = 0.0, None
        for d in deps[name]:
            f = visit(d)
            if f > start:
                start, best = f, d
        finish[name] = start + float(by_name[name]["days"])
        prev[name] = best
        state[name] = "done"
        return finish[name]

    for n in names:
        visit(n)
    end = max(names, key=lambda n: finish[n])
    path = []
    node = end
    while node:
        path.append(node)
        node = prev[node]
    return list(reversed(path)), finish[end]


REQUIREMENT_PATTERNS = {
    "metric": r"\b(ttft|tpot|itl|latency|throughput|tokens?/s|tokens per second|accuracy|e2e|time to first token)\b",
    "percentile": r"\b(p50|p90|p95|p99|median|percentile)\b",
    "threshold": r"(≤|<=|<|under|below|at most|no more than|≥|>=|at least)\s*\d|\d+(\.\d+)?\s*(ms|s|seconds|tokens|%|points)",
    "load": r"\b(\d+\s*(concurrent|users|requests|rps|qps|sequences)|concurrency)\b",
    "inputs": r"\b(\d[\d,]*\s*-?\s*tokens?\s*(inputs?|in|prompts?|outputs?|out)|inputs?|outputs?|prompt length|context)\b",
    "configuration": r"\b(config|configuration|precision|bf16|fp16|int8|fp8|model|hardware|stack|version)\b",
}


def requirement_check(text: str) -> dict:
    """Rule-based check of which requirement elements appear in the text."""
    lowered = text.lower()
    return {name: bool(re.search(pattern, lowered)) for name, pattern in REQUIREMENT_PATTERNS.items()}


def update_check(text: str) -> dict:
    """Rule-based checks for an executive bottom line."""
    lowered = text.lower()
    words = len(re.findall(r"\b\w[\w'.-]*\b", text))
    return {
        "words": words,
        "short_enough": 0 < words <= 30,
        "has_status": bool(re.search(r"\b(on track|at risk|off track|blocked|green|yellow|amber|red|go|no-go)\b", lowered)),
        "has_reason": bool(re.search(r"\b(because|due to|since|but|as)\b|[-–:]", lowered)),
        "has_ask": bool(re.search(r"\b(need|needs|ask|request|decide|decision|approve|help)\b", lowered)),
    }


def rough_token_count(text: str) -> int:
    """VERY rough token estimate: words and punctuation, with long words split every 6 characters."""
    pieces = re.findall(r"\w+|[^\w\s]", text)
    return sum(max(1, math.ceil(len(p) / 6)) if p.isalnum() else 1 for p in pieces)


# ---------- quiz & diagnostic scoring ----------

def score_answers(questions: list[dict], answers: list[int | None]) -> dict:
    """Score multiple-choice answers. Unanswered (None) counts as wrong."""
    if len(questions) != len(answers):
        raise ValueError("answers must match questions")
    results = [a is not None and a == q["answer"] for q, a in zip(questions, answers)]
    return {"correct": sum(results), "total": len(results), "results": results,
            "pct": round(100 * sum(results) / len(results)) if results else 0}


def diagnostic_by_topic(questions: list[dict], answers: list[int | None]) -> dict:
    """Per-topic correct/total from the diagnostic."""
    scored = score_answers(questions, answers)["results"]
    out: dict[str, dict] = {}
    for q, ok in zip(questions, scored):
        t = out.setdefault(q["topic"], {"correct": 0, "total": 0})
        t["total"] += 1
        t["correct"] += int(ok)
    return out


def recommend_path(by_topic: dict) -> str:
    """beginner if foundations are shaky; senior if everything is strong; otherwise working."""
    def ratio(topic):
        t = by_topic.get(topic, {"correct": 0, "total": 1})
        return t["correct"] / t["total"] if t["total"] else 0

    if ratio("foundations") < 2 / 3:
        return "beginner"
    middle = [ratio(t) for t in ("intel-judgment", "customer-engineering", "performance")]
    if min(middle) >= 2 / 3 and ratio("senior-tpm") >= 2 / 3:
        return "senior"
    return "working"
