# Risk and dependency register

> Fictional example.

| ID | Type | Description | Likelihood (1-5) | Impact (1-5) | Owner | Mitigation | Trigger | Status |
|---|---|---|---|---|---|---|---|---|
| R1 | Risk | Customer dataset approval delayed | 4 | 5 | Customer legal liaison | Use public stand-in dataset with matching lengths | No approval by 2026-11-06 | Open |
| R2 | Risk | Rack power insufficient for full config | 3 | 4 | Customer facilities | Confirm in writing; plan reduced config | Facilities can't confirm by ship date | Mitigating |
| D1 | Dependency | Serving fix for long-context hang | - | 5 | Software team lead | Track in weekly sync; test release candidate | Fix not in RC by 2026-11-20 | Open |
| D2 | Dependency | OEM firmware update | - | 3 | OEM contact | Apply in maintenance window | Firmware not released by 2026-11-13 | On track |
| R3 | Risk | INT8 fails accuracy gate | 3 | 2 | ML engineer | Keep BF16 as baseline config | Eval below tolerance | Realized - accepted |

**Priority rule used:** critical-path items first, then impact × evidence confidence.
