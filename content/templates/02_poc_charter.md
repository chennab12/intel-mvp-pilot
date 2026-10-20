# POC charter

> Fictional example: **Woodgrove Health** (not a real customer).

**POC name:** Clinical notes summarizer on Intel · **Dates:** 2026-11-02 to 2026-11-27 · **Version:** 1.0

## Objective
Determine whether the proposed configuration meets Woodgrove's latency, accuracy and stability requirements for production rollout in Q1.

## Scope
- In scope: Inference for one 8B model, BF16; serving with vLLM; 2,000-token inputs, 256-token outputs.
- Out of scope: Training, fine-tuning, multi-site deployment.

## Acceptance criteria (signed off by customer)
| # | Criterion | Method | Owner |
|---|---|---|---|
| 1 | P95 TTFT ≤ 1.0 s at 40 concurrent users | Load test, 3 runs, warm-up excluded | Perf engineer |
| 2 | Accuracy within 1 point of reference | Agreed public eval set | ML engineer |
| 3 | 72-hour soak, no crashes | Continuous load at 50% peak | SRE |
| 4 | Cost per 1M tokens ≤ customer target | Cost model with measured throughput | TPM |

## Environment
See Environment inventory v1 (attached to every result).

## Team and RACI
| Role | Name (fictional) | Responsibility |
|---|---|---|
| TPM | You | Plan, risks, reporting |
| Customer technical owner | Dana Park | Environment, data approval |
| Intel CE engineer | Alex Rivera | Install, tuning |
| OEM contact | Sam Lee | Hardware, firmware |

## Milestones
1. Hardware ready and inventory captured (week 1)
2. Smoke + correctness + accuracy pass (week 2)
3. Performance and soak (weeks 3-4)
4. Readout and go/no-go (end of week 4)

## Risks (top 3)
See Risk and dependency register.
