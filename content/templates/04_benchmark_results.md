# Benchmark results

> Fictional, illustrative numbers. NOT measured on any real system.

**Run ID:** bench-2026-11-16-a · **Inventory:** v2 · **Data label:** SIMULATED EXAMPLE

## Workload
| Field | Value |
|---|---|
| Model / revision | Open 8B model, rev abc123 (fictional) |
| Precision | BF16 weights and KV cache |
| Input tokens | fixed 2,000 |
| Output tokens | fixed 256 |
| Concurrency | 8, 16, 32, 40, 64 |
| Warm-up | 50 requests discarded |
| Measured requests | 500 per point, 3 repetitions |
| Client | Same rack, 1 load generator |

## Results (fictional)
| Concurrency | P50 TTFT (s) | P95 TTFT (s) | P95 TPOT (ms) | Output tok/s (aggregate) | Accuracy gate |
|---|---|---|---|---|---|
| 8 | 0.21 | 0.30 | 24 | 320 | Pass |
| 16 | 0.28 | 0.41 | 27 | 590 | Pass |
| 32 | 0.47 | 0.74 | 33 | 960 | Pass |
| 40 | 0.58 | 0.93 | 36 | 1,110 | Pass |
| 64 | 1.10 | 1.85 | 48 | 1,330 | Pass |

## Interpretation
- Acceptance criterion P95 TTFT ≤ 1.0 s is met up to 40 concurrent users (0.93 s).
- Report throughput at the operating point (40 users), not the peak (64 users).

## Fairness checklist
- [x] Same model revision, precision, lengths and concurrency across compared systems
- [x] Warm-up excluded on all systems
- [x] Both systems pass the same accuracy gate
- [x] Inventories attached for all systems
