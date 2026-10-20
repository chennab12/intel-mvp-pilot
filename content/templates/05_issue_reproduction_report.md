# Issue reproduction report

> Fictional example. Use only public or synthetic data.

**Title:** Intermittent hang during long-context generation · **ID:** POC-123 (fictional) · **Severity:** High (blocks acceptance criterion 3)

## Environment
Inventory v2 attached (all layers). Reproduced on lab system with identical versions: **Yes**.

## Steps to reproduce (MRE)
1. Start the server with the attached config (max context 16K).
2. Run `repro.py` (30 lines, attached): 8 concurrent requests, 12,000-token inputs from a public dataset.
3. Repeat 20 times.

## Expected vs actual
- Expected: all requests complete.
- Actual: in ~1 of 20 runs, requests stop producing tokens after ~3 minutes; no error logged.

## Frequency and conditions
- 5 hangs in 100 runs. Not seen below 8K context.

## Evidence
- Server and driver logs (attached, timestamps marked)
- Device memory trace during hang

## What has been tried
- Lower concurrency to 4: no hang in 100 runs.
- Newer serving release: not yet tested.

## Ask
Engineering triage to identify component owner; target decision by 2026-11-20.
