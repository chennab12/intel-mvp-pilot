"""Safe, predefined lab runners.

Each runner takes a small dict of validated parameters and returns a result dict:
    {"kind": "measured" | "computed" | "simulated", "metrics": {...}, "table": [...], "notes": [...]}
No runner executes user-provided code or shell commands. Work sizes are capped so a
hosted app stays responsive.
"""
from __future__ import annotations

import os
import platform
import sys
import time
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from utils import calculators as calc


def system_inventory(_: dict | None = None) -> dict:
    return {
        "kind": "measured",
        "metrics": {
            "Operating system": platform.system(),
            "OS release": platform.release(),
            "Machine architecture": platform.machine(),
            "Processor string": platform.processor() or "(not reported by this OS)",
            "Logical CPUs": os.cpu_count(),
        },
        "notes": ["Measured from the machine running this app. On a hosted copy this is the hosting server, not your laptop."],
    }


INVENTORY_PACKAGES = ("numpy", "pandas", "streamlit", "plotly", "PyYAML")


def environment_inventory(_: dict | None = None) -> dict:
    rows = [{"layer": "OS", "item": "platform", "value": platform.platform()},
            {"layer": "Runtime", "item": "Python", "value": sys.version.split()[0]}]
    for pkg in INVENTORY_PACKAGES:
        try:
            ver = version(pkg)
        except PackageNotFoundError:
            ver = "not installed"
        rows.append({"layer": "Framework/app", "item": pkg, "value": ver})
    return {"kind": "measured", "table": rows,
            "notes": ["Add hardware, firmware, driver, runtime and container rows on a real POC system."]}


def bandwidth_vs_compute(params: dict) -> dict:
    mb = int(min(max(params.get("array_mb", 100), 10), 400))
    n = mb * 1024 * 1024 // 4
    a = np.ones(n, dtype=np.float32)
    a.copy()  # warm-up
    best_copy = min(_time(lambda: a.copy()) for _ in range(3))
    size = int(min(max(params.get("matrix", 1024), 256), 2048))
    m = np.random.default_rng(0).random((size, size), dtype=np.float32)
    m @ m  # warm-up
    best_mm = min(_time(lambda: m @ m) for _ in range(3))
    copy_gbs = 2 * a.nbytes / 1e9 / best_copy
    gflops = 2 * size ** 3 / 1e9 / best_mm
    return {
        "kind": "measured",
        "metrics": {
            "Array copy (read + write)": f"{2 * a.nbytes / 1e9:.2f} GB in {best_copy * 1000:.1f} ms",
            "Copy bandwidth": f"{copy_gbs:.1f} GB/s",
            "Matrix multiply": f"{size}x{size} in {best_mm * 1000:.1f} ms",
            "Matmul compute": f"{gflops:.1f} GFLOP/s",
            "Matmul arithmetic intensity (approx.)": f"{2 * size ** 3 / (3 * size * size * 4):.0f} FLOP/byte",
        },
        "notes": ["Best of 3 runs after one warm-up. Measured on the machine running this app."],
    }


def _time(fn) -> float:
    t = time.perf_counter()
    fn()
    return max(time.perf_counter() - t, 1e-9)


def quantization_error(params: dict) -> dict:
    per_channel = bool(params.get("per_channel"))
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 1000)).astype(np.float32)  # same 100,000 values as the lesson example
    if params.get("outliers"):
        x[:, :5] *= 8  # a few outlier channels make per-tensor scaling worse
    rows = [{"format": "FP32", "bytes": int(x.nbytes), "mean_abs_error": 0.0}]
    for name, levels, bits in (("INT8", 127, 8), ("INT4-style (16 levels)", 7, 4)):
        if per_channel:
            scale = np.abs(x).max(axis=0, keepdims=True) / levels
        else:
            scale = np.abs(x).max() / levels
        scale = np.where(scale == 0, 1.0, scale)
        q = np.clip(np.round(x / scale), -levels - 1, levels)
        err = float(np.abs(q * scale - x).mean())
        rows.append({"format": name, "bytes": int(x.size * bits / 8), "mean_abs_error": round(err, 5)})
    return {"kind": "computed", "table": rows,
            "notes": ["Fixed random data (seed 0)" + (" with 5 outlier channels." if params.get("outliers") else "."),
                      "Per-channel scaling usually lowers error when value ranges differ across channels."]}


def quantization_error_simple() -> float:
    """The exact example printed in the lesson text (per-tensor INT8, seed 0, 100k normal values)."""
    x = np.random.default_rng(0).normal(size=100_000).astype(np.float32)
    scale = np.abs(x).max() / 127
    q = np.round(x / scale).astype(np.int8)
    return float(np.abs(q.astype(np.float32) * scale - x).mean())


def token_counting(params: dict) -> dict:
    text = str(params.get("text", ""))[:20000]
    chunks = int(min(max(params.get("chunks", 0), 0), 20))
    chunk_tokens = int(min(max(params.get("chunk_tokens", 300), 50), 2000))
    base = calc.rough_token_count(text)
    total = base + chunks * chunk_tokens
    return {"kind": "computed",
            "metrics": {"Your text (estimate)": f"{base} tokens", "Retrieved chunks": f"{chunks} x {chunk_tokens} tokens",
                        "Prompt total (estimate)": f"{total} tokens"},
            "notes": ["Rough word-piece estimate, not a real tokenizer. Use the model's tokenizer for real numbers."]}


def _work(n: int) -> int:
    return sum(i * i for i in range(n))


def latency_percentiles(params: dict) -> dict:
    runs = int(min(max(params.get("runs", 200), 20), 1000))
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        _work(20000)
        times.append((time.perf_counter() - t) * 1000)
    s = calc.latency_summary(times)
    return {"kind": "measured", "samples_ms": times,
            "metrics": {k: (f"{v:.3f} ms" if k != "count" else v) for k, v in s.items()},
            "notes": ["Measured on the machine running this app. Includes the first (warm-up) runs on purpose."]}


def warmup(_: dict | None = None) -> dict:
    @lru_cache(maxsize=None)
    def table(n):
        return tuple(i * i for i in range(n))

    times = []
    for _ in range(5):
        t = time.perf_counter()
        table(1_000_000)
        times.append((time.perf_counter() - t) * 1000)
    steady = sorted(times[1:])[len(times[1:]) // 2]
    return {"kind": "measured", "samples_ms": times,
            "table": [{"call": i + 1, "ms": round(v, 3)} for i, v in enumerate(times)],
            "metrics": {"First call": f"{times[0]:.2f} ms", "Steady-state median (calls 2-5)": f"{steady:.4f} ms"},
            "notes": ["The first call builds a table (like compilation or allocation); later calls reuse it."]}


def pipeline_bottleneck(params: dict) -> dict:
    sizes = {k: int(min(max(params.get(k, v), 1000), 400_000)) for k, v in
             (("preprocess", 50_000), ("compute", 200_000), ("postprocess", 20_000))}
    rows, total = [], 0.0
    for stage, n in sizes.items():
        t = time.perf_counter()
        _work(n)
        dt = (time.perf_counter() - t) * 1000
        total += dt
        rows.append({"stage": stage, "work_size": n, "ms": round(dt, 3)})
    for r in rows:
        r["share_%"] = round(100 * r["ms"] / total, 1) if total else 0.0
    biggest = max(rows, key=lambda r: r["ms"])
    return {"kind": "measured", "table": rows,
            "metrics": {"Total": f"{total:.2f} ms", "Largest share": f"{biggest['stage']} ({biggest['share_%']}%)"},
            "notes": [f"Making '{biggest['stage']}' twice as fast saves at most about {biggest['share_%'] / 2:.0f}% of total time."]}


def correctness_check(params: dict) -> dict:
    precision = params.get("precision", "FP16")
    tol = float(params.get("tolerance", 0.05))
    rng = np.random.default_rng(1)
    w, x = rng.normal(size=(256, 256)), rng.normal(size=256)
    ref = w @ x
    dtype = {"FP32": np.float32, "FP16": np.float16}[precision]
    cand = (w.astype(dtype) @ x.astype(dtype)).astype(np.float64)
    diff = float(np.abs(ref - cand).max())
    return {"kind": "computed",
            "metrics": {"Candidate precision": precision, "Max absolute difference": f"{diff:.6f}",
                        "Tolerance": f"{tol}", "Result": "PASS" if diff <= tol else "FAIL"},
            "notes": ["Reference is FP64 on fixed inputs (seed 1)."]}


# The "script" for the MRE lab: fixed steps, each mapped to a safe function.
MRE_STEPS = [
    ("load_config", "config = load_config('serve.yaml')"),
    ("setup_logging", "setup_logging(level='INFO')"),
    ("load_tokenizer", "tok = load_tokenizer('public-model')"),
    ("set_precision", "dtype = 'fp16'"),
    ("long_input", "prompt = make_prompt(tokens=12000)"),
    ("warm_cache", "warm_cache()"),
    ("concurrency", "requests = [prompt] * 8"),
    ("metrics_server", "start_metrics_server(port=9100)"),
    ("generate", "outputs = generate(requests, max_new_tokens=256)"),
    ("save", "save(outputs, 'out.jsonl')"),
]
MRE_TRIGGER = {"long_input", "concurrency", "generate"}


def shrink_mre(params: dict) -> dict:
    kept = [s for s, _ in MRE_STEPS if s in set(params.get("kept", []))]
    fails = MRE_TRIGGER.issubset(kept)
    minimal = set(kept) == MRE_TRIGGER
    msg = "FAIL - the hang still reproduces" if fails else "PASS - no hang; you removed a step the bug needs"
    notes = [f"{len(kept)} of {len(MRE_STEPS)} steps kept."]
    if fails and minimal:
        notes.append("This is the minimal reproducer: long input + concurrency + generate.")
    elif fails:
        notes.append("Still failing. Can you remove more steps?")
    return {"kind": "computed", "metrics": {"Result": msg}, "notes": notes, "minimal": fails and minimal,
            "fails": fails}


DEFAULT_PLAN = [
    {"task": "Discovery", "days": 5, "depends_on": ""},
    {"task": "Hardware ship", "days": 15, "depends_on": "Discovery"},
    {"task": "Security review", "days": 10, "depends_on": "Discovery"},
    {"task": "Install + smoke", "days": 5, "depends_on": "Hardware ship, Security review"},
    {"task": "Accuracy", "days": 5, "depends_on": "Install + smoke"},
    {"task": "Benchmark", "days": 5, "depends_on": "Accuracy"},
    {"task": "Readout", "days": 1, "depends_on": "Benchmark"},
]


def critical_path(params: dict) -> dict:
    tasks = params.get("tasks") or DEFAULT_PLAN
    path, total = calc.critical_path(tasks)
    return {"kind": "computed", "metrics": {"Critical path": " -> ".join(path), "Total duration": f"{total:g} days"},
            "notes": ["Tasks not on this path have slack."]}


def requirement_check(params: dict) -> dict:
    res = calc.requirement_check(str(params.get("text", "")))
    return {"kind": "computed", "table": [{"element": k, "present": "yes" if v else "missing"} for k, v in res.items()],
            "metrics": {"Elements present": f"{sum(res.values())} of {len(res)}"},
            "notes": ["Rule-based keyword check, not an AI judgment. A human still has to agree it's testable."]}


def update_checker(params: dict) -> dict:
    res = calc.update_check(str(params.get("text", "")))
    return {"kind": "computed",
            "metrics": {"Words": res["words"], "Under 30 words": "yes" if res["short_enough"] else "no",
                        "Status word": "yes" if res["has_status"] else "missing",
                        "Reason": "yes" if res["has_reason"] else "missing",
                        "Ask": "yes" if res["has_ask"] else "missing"},
            "notes": ["Rule-based check."]}


CLASSIFY_ITEMS = [
    ("Answer customer questions with an existing open model and our FAQ documents.", "Inference",
     "Using a trained model (plus retrieval) to answer requests is inference."),
    ("Teach an existing model our product catalog style with 5,000 examples.", "Fine-tuning",
     "Adapting an existing model with a smaller dataset is fine-tuning."),
    ("Build a brand-new foundation model from a large web-scale dataset.", "Training",
     "Learning all parameters from scratch is training."),
    ("Check whether the quantized model still meets our accuracy bar.", "Evaluation",
     "Measuring quality on agreed data is evaluation."),
    ("Summarize 10,000 call transcripts overnight with an existing model.", "Inference",
     "Batch (offline) use of a trained model is still inference."),
]
CLASSIFY_LABELS = ["Training", "Fine-tuning", "Inference", "Evaluation"]


def classify_workload(params: dict) -> dict:
    answers = params.get("answers", [])
    rows, correct = [], 0
    for i, (req, label, why) in enumerate(CLASSIFY_ITEMS):
        given = answers[i] if i < len(answers) else None
        ok = given == label
        correct += ok
        rows.append({"request": req, "your answer": given or "-", "answer": label, "ok": "yes" if ok else "no", "why": why})
    return {"kind": "computed", "table": rows, "metrics": {"Score": f"{correct} / {len(CLASSIFY_ITEMS)}"}}


def scaling_efficiency(params: dict) -> dict:
    comm = float(min(max(params.get("comm", 0.1), 0.0), 0.9))
    devices = int(min(max(params.get("devices", 8), 1), 64))
    rows = [{"devices": n, "speedup": round(calc.scaling_speedup(n, comm), 2),
             "efficiency_%": round(100 * calc.scaling_speedup(n, comm) / n, 1)} for n in range(1, devices + 1)]
    return {"kind": "simulated", "table": rows,
            "metrics": {f"Speed-up at {devices} devices": f"{rows[-1]['speedup']}x",
                        "Scaling efficiency": f"{rows[-1]['efficiency_%']}%"},
            "notes": ["Toy model for teaching; real systems overlap communication with compute."]}


RUNNERS = {
    "system_inventory": system_inventory,
    "environment_inventory": environment_inventory,
    "bandwidth_vs_compute": bandwidth_vs_compute,
    "quantization_error": quantization_error,
    "token_counting": token_counting,
    "latency_percentiles": latency_percentiles,
    "warmup": warmup,
    "pipeline_bottleneck": pipeline_bottleneck,
    "correctness_check": correctness_check,
    "shrink_mre": shrink_mre,
    "critical_path": critical_path,
    "requirement_check": requirement_check,
    "update_checker": update_checker,
    "classify_workload": classify_workload,
    "scaling_efficiency": scaling_efficiency,
}
