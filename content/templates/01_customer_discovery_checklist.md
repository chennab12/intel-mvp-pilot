# Customer discovery checklist

> Fictional example: **Fourth Coffee** (not a real customer). Replace every value with your own.

**Date:** 2026-10-20 · **TPM:** (your name) · **Account team:** (names)

## 1. Business goal and decision
- [x] Business goal: Answer barista and store-manager questions about recipes and equipment in under 2 seconds.
- [x] Decision this POC informs: Choose the on-prem inference platform for FY27 rollout.
- [x] Economic buyer: VP Store Technology (fictional). Technical owner: Platform engineering lead (fictional).
- [ ] Decision date: _open question, owner: account lead, due 2026-10-27_

## 2. Workload
- [x] Task: Inference with retrieval (RAG). No training in scope.
- [x] Model: Open 8B-parameter instruction model; BF16 acceptable; INT8 only if accuracy passes.
- [x] Input tokens: median 1,800, P95 3,200 (includes retrieved chunks).
- [x] Output tokens: median 180, P95 400.
- [x] Traffic: 200 concurrent users at lunch peak; 20 off-peak.

## 3. Targets
- [x] P95 TTFT ≤ 2.0 s at peak load.
- [x] Accuracy: within 1 point of reference on the agreed public evaluation set.
- [ ] Cost target: _pending finance input_

## 4. Environment
- [x] On-prem Kubernetes, Ubuntu LTS, air-cooled racks.
- [x] Security: no customer data leaves site; public stand-in dataset for POC.
- [ ] Power per rack: _facilities to confirm_

## 5. After success
- [x] Production handoff owner: Platform engineering.
- [ ] Support path agreed: _open_

## Open questions
| Question | Owner | Due |
|---|---|---|
| Decision date | Account lead | 2026-10-27 |
| Rack power budget | Customer facilities | 2026-10-24 |
