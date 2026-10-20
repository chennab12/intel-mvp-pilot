# Release-readiness checklist

> Fictional release: serving stack update for Woodgrove Health.

| Gate | Status (Pass / At risk / Fail / Not started) | Evidence | Owner |
|---|---|---|---|
| Supported combination published (OS, firmware, driver, framework, serving) | Pass | Matrix diff link | Release manager |
| Install from scratch using published docs | Pass | Clean-install log | QA |
| Functional tests on release candidate | Pass | Test report RC2 | QA |
| Accuracy within tolerance on RC | Pass | Eval report RC2 | ML engineer |
| Performance at customer operating point | At risk | P95 8% over target at peak | Perf engineer |
| 72-hour soak | Not started | - | SRE |
| Known issues documented with workarounds | Pass | Release notes draft | TPM |
| Support team briefed | Pass | Briefing notes | Support lead |
| Rollback plan tested | Not started | - | SRE |

**Recommendation:** Not ready. Re-assess after soak and rollback test (target 2026-12-04).
