"""A tiny but real ML model plus test harnesses for the Model Functional / Performance Testing sections.

- TicketClassifier: bag-of-words softmax regression in NumPy, trained on fictional POC tickets.
  It's small enough to train in well under a second on a laptop, and it's a real model:
  real weights, real inference, real measured latency.
- SimulatedLLMServer: a deterministic stand-in for an OpenAI-style completions endpoint, used to
  teach API-level functional checks. Its text and token timings are SIMULATED.
- simulate_load_test: a discrete-event simulation of a continuous-batching LLM server replaying
  a request trace. SIMULATED; not a model of any real product.
No Streamlit imports here so everything is unit-testable.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import math
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from utils import calculators as calc

DATA_DIR = Path(__file__).resolve().parent.parent / "content" / "model_lab"
LABELS = ["driver", "memory", "performance", "accuracy", "network"]
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


@lru_cache(maxsize=1)
def load_tickets() -> tuple[dict, ...]:
    with open(DATA_DIR / "tickets.csv", encoding="utf-8") as f:
        return tuple(csv.DictReader(f))


@lru_cache(maxsize=1)
def load_trace() -> tuple[dict, ...]:
    with open(DATA_DIR / "request_trace.csv", encoding="utf-8") as f:
        return tuple({"request_id": r["request_id"], "arrival_s": float(r["arrival_s"]),
                      "input_tokens": int(r["input_tokens"]), "output_tokens": int(r["output_tokens"])}
                     for r in csv.DictReader(f))


def split(rows, name):
    return [r for r in rows if r["split"] == name]


class TicketClassifier:
    """Multinomial logistic regression over a bag-of-words vocabulary."""

    def __init__(self, vocab: list[str], weights: np.ndarray, bias: np.ndarray, labels=LABELS):
        self.vocab = vocab
        self.index = {w: i for i, w in enumerate(vocab)}
        self.W = weights
        self.b = bias
        self.labels = list(labels)

    @classmethod
    def train(cls, rows, epochs: int = 300, lr: float = 0.5, l2: float = 1e-3, seed: int = 0) -> "TicketClassifier":
        vocab = sorted({tok for r in rows for tok in tokenize(r["text"])})
        model = cls(vocab, np.zeros((len(vocab), len(LABELS))), np.zeros(len(LABELS)))
        X = model.featurize([r["text"] for r in rows])
        y = np.array([LABELS.index(r["label"]) for r in rows])
        Y = np.eye(len(LABELS))[y]
        rng = np.random.default_rng(seed)
        model.W = rng.normal(0, 0.01, size=model.W.shape)
        for _ in range(epochs):
            P = model._softmax(X @ model.W + model.b)
            grad = (P - Y) / len(rows)
            model.W -= lr * (X.T @ grad + l2 * model.W)
            model.b -= lr * grad.sum(axis=0)
        return model

    def featurize(self, texts: list[str]) -> np.ndarray:
        X = np.zeros((len(texts), len(self.vocab)))
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                j = self.index.get(tok)
                if j is not None:
                    X[i, j] = 1.0
        return X

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self._softmax(self.featurize(texts) @ self.W + self.b)

    def predict(self, texts: list[str]) -> list[str]:
        return [self.labels[i] for i in self.predict_proba(texts).argmax(axis=1)]

    def checksum(self) -> str:
        return hashlib.sha256(self.W.astype(np.float32).tobytes() + self.b.astype(np.float32).tobytes()).hexdigest()[:16]

    def parameter_count(self) -> int:
        return int(self.W.size + self.b.size)

    def quantized(self) -> "TicketClassifier":
        """Per-tensor symmetric INT8 copy of the weights (dequantized for inference)."""
        scale = np.abs(self.W).max() / 127 or 1.0
        Wq = np.round(self.W / scale).clip(-128, 127) * scale
        return TicketClassifier(self.vocab, Wq, self.b.copy(), self.labels)


MODEL_FAULTS = {
    "None (healthy model)": None,
    "Label order swapped in the deployed config": "label_swap",
    "Weights corrupted by a bad conversion": "weight_noise",
    "Tokenizer mismatch (deployed tokenizer drops words)": "tokenizer",
}


def with_fault(model: TicketClassifier, fault: str | None) -> TicketClassifier:
    """Return a copy of the model with a classic deployment bug injected, so the tests have something to catch."""
    if fault is None:
        return model
    if fault == "label_swap":
        labels = list(model.labels)
        labels[0], labels[1] = labels[1], labels[0]
        return TicketClassifier(model.vocab, model.W, model.b, labels)
    if fault == "weight_noise":
        rng = np.random.default_rng(7)
        return TicketClassifier(model.vocab, model.W + rng.normal(0, 6 * np.abs(model.W).std(), model.W.shape),
                                model.b, model.labels)
    if fault == "tokenizer":
        keep = model.vocab[::3]  # the "deployed" tokenizer only knows a third of the vocabulary
        idx = [model.index[w] for w in keep]
        return TicketClassifier(keep, model.W[idx], model.b, model.labels)
    raise ValueError(f"unknown fault {fault}")


@lru_cache(maxsize=1)
def trained_model() -> TicketClassifier:
    return TicketClassifier.train(split(load_tickets(), "train"))


# Golden outputs: fixed inputs with the label a correct model must return. These are the
# "known answers" a functional test checks after any change (new hardware, precision, version).
GOLDEN = [
    ("device not detected after the driver upgrade", "driver"),
    ("out of memory at 64 users with long prompts", "memory"),
    ("p95 latency doubled and throughput dropped", "performance"),
    ("accuracy dropped after int8 quantization", "accuracy"),
    ("rdma link slow and all reduce stalls across nodes", "network"),
]

EDGE_CASES = [
    ("Empty input", ""),
    ("Only unknown words", "zzzz qqqq xyzzy"),
    ("Very long input (5,000 words)", "memory " * 5000),
    ("Non-ASCII and punctuation", "¡Error! «device» не найден ⚠️ ???"),
    ("Numbers only", "404 500 8192"),
]


def accuracy_report(model: TicketClassifier, rows) -> dict:
    texts = [r["text"] for r in rows]
    truth = [r["label"] for r in rows]
    pred = model.predict(texts)
    correct = sum(p == t for p, t in zip(pred, truth))
    confusion = {t: {p: 0 for p in LABELS} for t in LABELS}
    for t, p in zip(truth, pred):
        confusion[t][p] += 1
    per_class = {}
    for lab in LABELS:
        tp = confusion[lab][lab]
        fp = sum(confusion[t][lab] for t in LABELS if t != lab)
        fn = sum(confusion[lab][p] for p in LABELS if p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[lab] = {"precision": round(prec, 3), "recall": round(rec, 3), "support": tp + fn}
    errors = [{"id": r["id"], "text": r["text"], "expected": t, "predicted": p}
              for r, t, p in zip(rows, truth, pred) if t != p]
    return {"accuracy": correct / len(rows), "correct": correct, "total": len(rows), "confusion": confusion,
            "per_class": per_class, "errors": errors, "predictions": pred}


def run_functional_suite(model: TicketClassifier, threshold: float = 0.85, quant_tolerance_pts: float = 2.0) -> list[dict]:
    """Run every functional check and return one row per check with PASS/FAIL and evidence."""
    test = split(load_tickets(), "test")
    results = []

    def add(step, name, ok, evidence):
        results.append({"step": step, "check": name, "result": "PASS" if ok else "FAIL", "evidence": evidence})

    # 1. Load + inventory
    add(1, "Model loads and reports inventory", model.parameter_count() > 0,
        f"{model.parameter_count():,} parameters, vocab {len(model.vocab)}, labels {len(model.labels)}, sha256[:16]={model.checksum()}")
    # 2. Smoke test
    p = model.predict_proba(["device not detected"])[0]
    ok = p.shape == (len(LABELS),) and abs(p.sum() - 1) < 1e-9 and (p >= 0).all()
    add(2, "Smoke test: one request, valid output", ok, f"shape={p.shape}, sum={p.sum():.6f}, top={LABELS[int(p.argmax())]}")
    # 3. Determinism
    a = model.predict_proba([g[0] for g in GOLDEN])
    b = model.predict_proba([g[0] for g in GOLDEN])
    add(3, "Determinism: same input -> same output", np.array_equal(a, b), f"max diff {np.abs(a - b).max():.1e}")
    # 4. Golden outputs
    got = model.predict([g[0] for g in GOLDEN])
    misses = [f"'{g[0]}' -> {o} (want {g[1]})" for g, o in zip(GOLDEN, got) if o != g[1]]
    add(4, "Golden outputs match known answers", not misses, f"{len(GOLDEN) - len(misses)}/{len(GOLDEN)} match" +
        ("; " + "; ".join(misses) if misses else ""))
    # 5. Edge cases
    problems = []
    for name, text in EDGE_CASES:
        try:
            q = model.predict_proba([text])[0]
            if not np.isfinite(q).all() or abs(q.sum() - 1) > 1e-6:
                problems.append(name)
        except Exception as exc:  # noqa: BLE001 - a functional test must catch everything
            problems.append(f"{name}: {exc}")
    add(5, "Edge cases don't crash and stay valid", not problems,
        f"{len(EDGE_CASES) - len(problems)}/{len(EDGE_CASES)} handled" + (f"; failed: {problems}" if problems else ""))
    # 6. Accuracy gate
    rep = accuracy_report(model, test)
    add(6, f"Accuracy on held-out test set >= {threshold:.0%}", rep["accuracy"] >= threshold,
        f"{rep['correct']}/{rep['total']} = {rep['accuracy']:.1%}")
    # 7. Lower-precision regression
    qrep = accuracy_report(model.quantized(), test)
    agree = np.mean([x == y for x, y in zip(rep["predictions"], qrep["predictions"])])
    delta = 100 * (rep["accuracy"] - qrep["accuracy"])
    add(7, f"INT8 weights within {quant_tolerance_pts:g} pts of FP64 reference", delta <= quant_tolerance_pts,
        f"INT8 {qrep['accuracy']:.1%} vs reference {rep['accuracy']:.1%} (delta {delta:+.1f} pts), prediction agreement {agree:.0%}")
    return results


# ---------------- Simulated OpenAI-style endpoint (functional API checks) ----------------

class SimulatedLLMServer:
    """Deterministic fake completions endpoint. Responses are SIMULATED text, built from a fixed word list."""

    WORDS = ("the accelerator memory budget depends on weights kv cache and runtime overhead "
             "measure ttft and tpot at the agreed concurrency then compare with the target").split()
    model_name = "demo-llm-1b (simulated)"

    def __init__(self, max_context: int = 4096, fault: str | None = None):
        self.max_context = max_context
        self.fault = fault  # None | "ignore_max_tokens" | "nondeterministic" | "bad_schema"
        self._calls = 0

    def completions(self, prompt: str, max_tokens: int = 32, temperature: float = 0.0, stop: list[str] | None = None) -> dict:
        self._calls += 1
        prompt_tokens = calc.rough_token_count(prompt)
        if prompt_tokens + max_tokens > self.max_context:
            return {"error": {"type": "invalid_request_error", "message": "context length exceeded",
                              "prompt_tokens": prompt_tokens, "max_context": self.max_context}}
        seed = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 997
        if temperature > 0 or self.fault == "nondeterministic":
            seed += self._calls
        n = max_tokens * 2 if self.fault == "ignore_max_tokens" else max_tokens
        words = [self.WORDS[(seed + i * 7) % len(self.WORDS)] for i in range(n)]
        finish = "length"
        if stop:
            for s in stop:
                if s in words:
                    words = words[:words.index(s)]
                    finish = "stop"
                    break
        resp = {"id": f"cmpl-{self._calls}", "object": "text_completion", "model": self.model_name,
                "choices": [{"index": 0, "text": " ".join(words), "finish_reason": finish}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(words),
                          "total_tokens": prompt_tokens + len(words)}}
        if self.fault == "bad_schema":
            del resp["usage"]
        return resp


def run_api_checks(server: SimulatedLLMServer) -> list[dict]:
    rows = []

    def add(name, ok, evidence):
        rows.append({"check": name, "result": "PASS" if ok else "FAIL", "evidence": evidence})

    r = server.completions("Explain KV cache in one line.", max_tokens=16)
    schema_ok = "choices" in r and "usage" in r and r["choices"] and "text" in r["choices"][0]
    add("Response has the expected schema (choices, text, usage)", bool(schema_ok), ", ".join(sorted(r)))
    n = len(r["choices"][0]["text"].split()) if schema_ok else -1
    add("max_tokens is respected", schema_ok and n <= 16, f"completion words={n}, max_tokens=16")
    u = r.get("usage")
    add("usage totals add up", bool(u) and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"],
        str(u) if u else "usage field missing")
    r1 = server.completions("Same prompt", max_tokens=12, temperature=0)
    r2 = server.completions("Same prompt", max_tokens=12, temperature=0)
    same = r1.get("choices", [{}])[0].get("text") == r2.get("choices", [{}])[0].get("text")
    add("temperature=0 is deterministic", same, "two identical requests " + ("matched" if same else "differed"))
    r3 = server.completions("Stop test", max_tokens=30, stop=["measure"])
    txt = r3.get("choices", [{}])[0].get("text", "")
    add("stop sequence honored", "measure" not in txt.split(), f"finish_reason={r3.get('choices', [{}])[0].get('finish_reason')}")
    r4 = server.completions("x " * (server.max_context + 10), max_tokens=8)
    add("over-long prompt returns a clear error (no crash)", "error" in r4, str(r4.get("error", r4.get("object")))[:120])
    return rows


# ---------------- Performance: measured on the real tiny model ----------------

def measure_latency(model: TicketClassifier, requests: int = 300, warmup: int = 20) -> dict:
    texts = [r["text"] for r in load_tickets()]
    cold = []
    for i in range(warmup):
        t = time.perf_counter()
        model.predict_proba([texts[i % len(texts)]])
        cold.append((time.perf_counter() - t) * 1000)
    times = []
    for i in range(requests):
        t = time.perf_counter()
        model.predict_proba([texts[i % len(texts)]])
        times.append((time.perf_counter() - t) * 1000)
    return {"warmup_ms": cold, "samples_ms": times, "summary": calc.latency_summary(times),
            "first_call_ms": cold[0] if cold else None}


def measure_batch_scaling(model: TicketClassifier, batch_sizes=(1, 4, 16, 64, 256), repeats: int = 5) -> list[dict]:
    texts = [r["text"] for r in load_tickets()]
    rows = []
    for bs in batch_sizes:
        batch = [texts[i % len(texts)] for i in range(bs)]
        model.predict_proba(batch)  # warm-up
        best = math.inf
        for _ in range(repeats):
            t = time.perf_counter()
            model.predict_proba(batch)
            best = min(best, time.perf_counter() - t)
        rows.append({"batch_size": bs, "batch_latency_ms": round(best * 1000, 3),
                     "per_item_ms": round(best * 1000 / bs, 4), "items_per_s": round(bs / best)})
    return rows


# ---------------- Performance: simulated LLM load test replaying a trace ----------------

def simulate_load_test(trace, *, rate_multiplier: float = 1.0, max_batch: int = 32,
                       prefill_ms_per_1k: float = 60.0, step_base_ms: float = 18.0, step_per_seq_ms: float = 0.5,
                       max_requests: int | None = None) -> dict:
    """Event-driven SIMULATION of a continuous-batching server.

    Each scheduler step: admit queued requests up to max_batch (their prefill time is added to the step),
    then every running request decodes one token. Step time = base + per_seq x running + admitted prefill.
    """
    reqs = [dict(r, arrival_s=r["arrival_s"] / rate_multiplier) for r in trace]
    if max_requests:
        reqs = reqs[:max_requests]
    pending = list(reqs)
    queue, running, done = [], [], []
    now = 0.0
    while pending or queue or running:
        while pending and pending[0]["arrival_s"] <= now:
            queue.append(pending.pop(0))
        if not queue and not running:
            now = pending[0]["arrival_s"]
            continue
        admitted = []
        while queue and len(running) + len(admitted) < max_batch:
            admitted.append(queue.pop(0))
        prefill_ms = sum(prefill_ms_per_1k * r["input_tokens"] / 1000 for r in admitted)
        step_ms = step_base_ms + step_per_seq_ms * (len(running) + len(admitted)) + prefill_ms
        now += step_ms / 1000
        new_ids = {r["request_id"] for r in admitted}
        for r in admitted:
            r["first_token_s"] = now
            r["generated"] = 1
            running.append(r)
        still = []
        for r in running:
            if r["request_id"] not in new_ids:
                r["generated"] += 1
            if r["generated"] >= r["output_tokens"]:
                r["finish_s"] = now
                done.append(r)
            else:
                still.append(r)
        running = still
    ttft = [(r["first_token_s"] - r["arrival_s"]) * 1000 for r in done]
    e2e = [(r["finish_s"] - r["arrival_s"]) * 1000 for r in done]
    tpot = [((r["finish_s"] - r["first_token_s"]) * 1000) / max(1, r["output_tokens"] - 1) for r in done]
    duration = max(r["finish_s"] for r in done) - min(r["arrival_s"] for r in done)
    out_tokens = sum(r["output_tokens"] for r in done)
    return {
        "requests": len(done),
        "offered_rate_rps": round(len(done) / max(1e-9, max(r["arrival_s"] for r in done)), 2),
        "duration_s": round(duration, 2),
        "ttft_ms": calc.latency_summary(ttft),
        "tpot_ms": calc.latency_summary(tpot),
        "e2e_ms": calc.latency_summary(e2e),
        "output_tokens_per_s": round(out_tokens / duration, 1),
        "request_throughput_rps": round(len(done) / duration, 2),
        "per_request": [{"request_id": r["request_id"], "input_tokens": r["input_tokens"],
                         "output_tokens": r["output_tokens"], "ttft_ms": round(t, 1), "tpot_ms": round(p, 2),
                         "e2e_ms": round(e, 1)} for r, t, p, e in zip(done, ttft, tpot, e2e)],
    }


def goodput(per_request: list[dict], ttft_slo_ms: float, tpot_slo_ms: float) -> float:
    """Share of requests that met BOTH SLOs."""
    if not per_request:
        return 0.0
    ok = sum(1 for r in per_request if r["ttft_ms"] <= ttft_slo_ms and r["tpot_ms"] <= tpot_slo_ms)
    return ok / len(per_request)


def heap_top(values, k):
    return heapq.nlargest(k, values)
