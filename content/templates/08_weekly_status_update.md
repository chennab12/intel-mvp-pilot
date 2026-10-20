# Weekly technical status update

> Fictional example. Week of 2026-11-16 · Woodgrove Health POC

**Bottom line:** At risk - latency and accuracy pass, but a long-context hang blocks the soak test; need a fix in RC by Nov 20.

## Changed since last week
- Accuracy gate passed (BF16), report v2.
- P95 TTFT 0.93 s at 40 users (target ≤ 1.0 s), 3 runs.
- Hang reproduced with a 30-line public MRE (1 in 20 runs).

## Top risks / blockers
| Item | Owner | Next step | Date |
|---|---|---|---|
| Long-context hang (D1) | Software team lead | Root cause + RC fix | Nov 20 |
| Dataset approval (R1) | Customer legal liaison | Sign-off | Nov 18 |

## Decisions / help needed
- Engineering priority for D1 (critical path).

## Confidence labels
Latency and accuracy: measured. Cost: estimate (utilization assumed 70%).
