"""Regenerate the fictional test data used by the Model Testing and Agentic AI sections.

Usage:  python scripts/make_model_lab_data.py
Writes content/model_lab/tickets.csv and content/model_lab/request_trace.csv (deterministic, seed 42).
All text is fictional and synthetic.
"""
import csv
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "content" / "model_lab"

TEMPLATES = {
    "driver": [
        "after the {pkg} upgrade the device is not detected",
        "framework says no device found after reinstalling the {pkg} driver",
        "initialization error at startup since the {pkg} runtime update",
        "version mismatch between driver and {pkg} runtime breaks startup",
        "service account cannot open the device permission denied on render node",
    ],
    "memory": [
        "out of memory when concurrency reaches {n} users",
        "allocation failed for kv cache at {n}k context length",
        "model weights do not fit on the card at bf16",
        "oom error during long prompts with {n} sequences",
        "device memory climbs until the server crashes at {n} users",
    ],
    "performance": [
        "throughput is half of the benchmark on the {pkg} server",
        "p95 latency doubled after moving to the new cluster",
        "time to first token is too slow for {n} concurrent users",
        "tokens per second dropped after the {pkg} update",
        "utilization is low and the accelerator looks idle during serving",
    ],
    "accuracy": [
        "answers got worse after int8 quantization",
        "evaluation score dropped {n} points versus the reference",
        "outputs differ from the reference platform on the golden prompts",
        "citation accuracy regressed after switching precision",
        "model output is truncated and quality looks wrong",
    ],
    "network": [
        "two servers train only slightly faster than one",
        "rdma link negotiated a lower speed on one port",
        "scaling across nodes stalls in all reduce communication",
        "switch port errors during multi node training",
        "collective communication dominates step time at {n} nodes",
    ],
}
PKGS = ["framework", "serving", "gpu", "gaudi", "openvino", "container"]


def tickets(seed=42):
    rng = random.Random(seed)
    rows = []
    for label, temps in TEMPLATES.items():
        for i in range(16):
            text = rng.choice(temps).format(pkg=rng.choice(PKGS), n=rng.choice([8, 16, 32, 40, 64]))
            if rng.random() < 0.3:
                text = rng.choice(["urgent: ", "customer reports ", "poc blocker: ", ""]) + text
            rows.append({"text": text, "label": label})
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["split"] = "test" if i % 4 == 0 else "train"
        r["id"] = f"T{i + 1:03d}"
    return rows


def trace(seed=42, n=120):
    rng = random.Random(seed)
    t, rows = 0.0, []
    for i in range(n):
        t += rng.expovariate(4.0)  # ~4 requests/s at the trace's base rate
        rows.append({"request_id": f"R{i + 1:03d}", "arrival_s": round(t, 3),
                     "input_tokens": int(min(8000, max(50, rng.lognormvariate(6.9, 0.6)))),
                     "output_tokens": int(min(1024, max(16, rng.lognormvariate(5.0, 0.5))))})
    return rows


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    write(OUT / "tickets.csv", tickets())
    write(OUT / "request_trace.csv", trace())
    print("wrote", OUT)
