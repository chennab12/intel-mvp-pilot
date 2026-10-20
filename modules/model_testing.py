"""Model Functional Testing and Model Performance Testing sections.

Both run on a real (tiny) NumPy model trained in-app on fictional tickets, plus clearly labeled
simulations of an LLM endpoint and an LLM server under load. Nothing typed by the user is executed.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.ui import ACCENT, KIND_TEXT, page_header, render_sources
from utils import model_lab as ml

GREY = "#9AA5B1"
RED = "#C0392B"


def _result_table(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df["result"] = df["result"].map(lambda r: "✅ PASS" if r == "PASS" else "❌ FAIL")
    st.dataframe(df, hide_index=True, width="stretch")


def _illustrative(code: str, lang: str = "bash", note: str = "") -> None:
    st.caption("**Illustrative, not run by this app.** Check every command against the docs for your exact "
               "versions before using it on a real system." + (f" {note}" if note else ""))
    st.code(code, language=lang, wrap_lines=True)


# ============================== Functional testing ==============================

def render_functional() -> None:
    page_header("✅ Model Functional Testing",
                "Is the model **working correctly**, before anyone talks about speed? This section walks the "
                "functional test plan a Customer Engineering team runs after any change: new hardware, new driver, "
                "new precision, new framework version. It uses a **real tiny model** (a NumPy text classifier that "
                "routes fictional POC tickets to driver, memory, performance, accuracy or network) and a "
                "**simulated** OpenAI-style endpoint.")
    st.info("💻 **Runs on a normal laptop.** The model trains in-app in well under a second on 60 fictional "
            "tickets and is tested on 20 held-out tickets. All tickets are fictional test data.")
    tabs = st.tabs(["1 · Test plan", "2 · Run the suite", "3 · Accuracy deep-dive", "4 · Serving API checks",
                    "5 · Try a ticket", "6 · On a real stack"])
    with tabs[0]:
        _functional_plan()
    with tabs[1]:
        _functional_suite()
    with tabs[2]:
        _accuracy_deep_dive()
    with tabs[3]:
        _api_checks()
    with tabs[4]:
        _try_ticket()
    with tabs[5]:
        _functional_real_stack()


def _functional_plan() -> None:
    st.subheader("The functional test plan, in order")
    st.markdown("Each step only makes sense if the one before it passed. Stop at the first failure and fix it "
                "before moving on; a performance number from a model that fails step 4 is meaningless.")
    st.dataframe(pd.DataFrame([
        {"Step": 1, "Check": "Load + inventory", "Question it answers": "Did the right model load? Parameter count, "
         "file checksum and labels match the reference?", "Typical real failure": "Wrong checkpoint, partial download"},
        {"Step": 2, "Check": "Smoke test", "Question it answers": "Does one request return a well-formed output?",
         "Typical real failure": "Device not found, missing operator, import error"},
        {"Step": 3, "Check": "Determinism", "Question it answers": "Same input, same settings: same output?",
         "Typical real failure": "Non-deterministic kernels, sampling left on"},
        {"Step": 4, "Check": "Golden outputs", "Question it answers": "Do fixed inputs give the known answers?",
         "Typical real failure": "Label/config mismatch, tokenizer mismatch"},
        {"Step": 5, "Check": "Edge cases", "Question it answers": "Empty, very long, odd characters: no crash?",
         "Typical real failure": "Crash on long context, encoding errors"},
        {"Step": 6, "Check": "Accuracy gate", "Question it answers": "Quality on a held-out set above the agreed bar?",
         "Typical real failure": "Silent quality regression"},
        {"Step": 7, "Check": "Lower-precision regression", "Question it answers": "Does INT8/FP8 stay within the "
         "agreed tolerance of the reference?", "Typical real failure": "Quantization outliers"},
    ]), hide_index=True, width="stretch")
    st.markdown("**TPM habit:** agree the accuracy bar, the tolerance and the golden set with the customer "
                "**before** the run, in writing. Otherwise every result becomes a negotiation.")


def _functional_suite() -> None:
    st.subheader("Run all seven checks on the real model")
    a, b, c = st.columns(3)
    threshold = a.slider("Accuracy bar (step 6)", 0.50, 1.00, 0.85, 0.05, key="mf_threshold")
    tol = b.slider("INT8 tolerance, points (step 7)", 0.0, 10.0, 2.0, 0.5, key="mf_tol")
    fault_name = c.selectbox("Inject a deployment bug", list(ml.MODEL_FAULTS), key="mf_fault",
                             help="Pick a classic real-world bug and see which checks catch it.")
    ref = ml.trained_model()
    model = ml.with_fault(ref, ml.MODEL_FAULTS[fault_name])
    rows = ml.run_functional_suite(model, threshold=threshold, quant_tolerance_pts=tol)
    passed = sum(r["result"] == "PASS" for r in rows)
    m1, m2, m3 = st.columns(3)
    m1.metric("Checks passed", f"{passed} / {len(rows)}")
    m2.metric("Reference checksum", ref.checksum())
    m3.metric("Deployed checksum", model.checksum(),
              delta="matches" if model.checksum() == ref.checksum() else "differs", delta_color="off")
    st.markdown(KIND_TEXT["measured"])
    _result_table(rows)
    if passed == len(rows):
        st.success("All functional checks pass. Only now is it worth measuring performance.")
    else:
        first = next(r for r in rows if r["result"] == "FAIL")
        st.error(f"First failure at step {first['step']}: {first['check']}. Stop and fix this before any "
                 "performance work.")
    st.markdown("**What to notice:**")
    st.markdown("- *Weights corrupted* still passes the accuracy gate here: only the golden outputs and the "
                "checksum catch it. One metric is never enough.\n"
                "- *Label order swapped* passes smoke and determinism tests: the model runs fine and is consistently "
                "wrong.\n- A checksum mismatch alone tells you the deployed artifact differs from the reference before "
                "you run a single request.")


def _accuracy_deep_dive() -> None:
    st.subheader("Where is the model right and wrong?")
    fault_name = st.selectbox("Model", list(ml.MODEL_FAULTS), key="mf_dd_fault")
    model = ml.with_fault(ml.trained_model(), ml.MODEL_FAULTS[fault_name])
    rep = ml.accuracy_report(model, ml.split(ml.load_tickets(), "test"))
    st.metric("Held-out accuracy", f"{rep['accuracy']:.0%}", help=f"{rep['correct']} of {rep['total']} tickets")
    z = [[rep["confusion"][t][p] for p in ml.LABELS] for t in ml.LABELS]
    fig = go.Figure(go.Heatmap(z=z, x=ml.LABELS, y=ml.LABELS, colorscale=[[0, "#F4F6F8"], [1, ACCENT]],
                               text=z, texttemplate="%{text}", showscale=False))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Predicted",
                      yaxis_title="Actual", yaxis_autorange="reversed", title="Confusion matrix (test set)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame([{"label": k, **v} for k, v in rep["per_class"].items()]), hide_index=True,
                 width="stretch")
    st.markdown("**Golden outputs and edge cases**")
    gold = model.predict([g[0] for g in ml.GOLDEN])
    st.dataframe(pd.DataFrame([{"input": g[0], "expected": g[1], "got": o, "ok": "✅" if o == g[1] else "❌"}
                               for g, o in zip(ml.GOLDEN, gold)]), hide_index=True, width="stretch")
    probs = model.predict_proba([e[1] for e in ml.EDGE_CASES])
    st.dataframe(pd.DataFrame([{"edge case": name, "prediction": model.labels[int(p.argmax())],
                                "top probability": round(float(p.max()), 3)}
                               for (name, _), p in zip(ml.EDGE_CASES, probs)]), hide_index=True, width="stretch")
    st.caption("An input with no known words falls back to the model's prior (low top probability). That's valid "
               "behaviour, but a real product might also return 'unknown': agree which, up front.")
    if rep["errors"]:
        st.markdown("**Misclassified test tickets**")
        st.dataframe(pd.DataFrame(rep["errors"]), hide_index=True, width="stretch")


def _api_checks() -> None:
    st.subheader("Functional checks on a serving endpoint")
    st.markdown("Once a model is behind an API, test the **contract**, not just the answers. This endpoint is "
                "**simulated**: it returns OpenAI-style JSON built from a fixed word list, so the checks behave like "
                "they would against a real server.")
    faults = {"None (healthy server)": None, "Server ignores max_tokens": "ignore_max_tokens",
              "Server is non-deterministic at temperature 0": "nondeterministic",
              "Response is missing the usage field": "bad_schema"}
    fault = st.selectbox("Inject a server bug", list(faults), key="mf_api_fault")
    server = ml.SimulatedLLMServer(fault=faults[fault])
    rows = ml.run_api_checks(server)
    st.markdown(KIND_TEXT["simulated"])
    _result_table(rows)
    with st.expander("See one raw response"):
        st.json(ml.SimulatedLLMServer(fault=faults[fault]).completions("Explain KV cache in one line.", max_tokens=16))


def _try_ticket() -> None:
    st.subheader("Classify your own ticket text")
    st.caption("Your text is only split into words and scored by the model. It is never executed or stored. "
               "Don't paste confidential information.")
    text = st.text_area("Ticket text", "after the firmware update the device is not detected by the runtime",
                        max_chars=1000, key="mf_ticket")
    model = ml.trained_model()
    p = model.predict_proba([text])[0]
    known = [t for t in ml.tokenize(text) if t in model.index]
    fig = go.Figure(go.Bar(x=ml.LABELS, y=p, marker_color=[ACCENT if i == p.argmax() else GREY for i in range(len(p))],
                           text=[f"{v:.0%}" for v in p], textposition="outside"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="probability", yaxis_range=[0, 1.1])
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"Prediction: **{ml.LABELS[int(p.argmax())]}**. Words the model knows: "
                f"{', '.join(known) if known else 'none (so it falls back to its prior)'}.")


def _functional_real_stack() -> None:
    st.subheader("The same steps on a real Intel stack")
    st.markdown("This app can't reach your hardware. Here is what each step usually looks like on a real system. "
                "Run these yourself, on a system you're authorized to use.")
    st.markdown("**Step 1–2: device visible to the framework, one tiny op runs** (PyTorch with Intel GPU support)")
    _illustrative('python -c "import torch; print(torch.__version__); print(torch.xpu.is_available()); '
                  'x = torch.ones(2, 2, device=\'xpu\'); print((x @ x).cpu())"', note="Requires a PyTorch build with XPU support.")
    st.markdown("**Step 2: smoke test of an OpenAI-compatible server** (for example vLLM)")
    _illustrative('curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" \\\n'
                  '  -d \'{"model": "<your-model>", "prompt": "Say hello", "max_tokens": 8, "temperature": 0}\'')
    st.markdown("**Steps 3–4: determinism and golden outputs, as a script**")
    _illustrative('''import requests

GOLDEN = [("What is 2 + 2? Answer with a number.", "4")]
URL = "http://localhost:8000/v1/completions"

for prompt, expected in GOLDEN:
    outs = [requests.post(URL, json={"model": "<your-model>", "prompt": prompt,
                                     "max_tokens": 4, "temperature": 0}).json()["choices"][0]["text"]
            for _ in range(2)]
    print("deterministic:", outs[0] == outs[1], "| golden:", expected in outs[0])''', lang="python")
    st.markdown("**Steps 6–7: accuracy on a public benchmark, reference vs lower precision.** Run the same "
                "evaluation harness and task list on the reference precision and on the quantized model, then "
                "compare against the tolerance agreed with the customer.")
    render_sources(["pytorch-xpu-get-started", "vllm-gpu-install"])


# ============================== Performance testing ==============================

def render_performance() -> None:
    page_header("⏱️ Model Performance Testing",
                "Is the model **fast enough, at the agreed load, for the agreed cost**? Only test performance after "
                "the functional checks pass. Part A **measures** the real tiny model on the machine running this "
                "app. Part B **simulates** an LLM server replaying a request trace so you can practise the "
                "metrics that matter in LLM POCs: TTFT, TPOT, throughput and goodput.")
    tabs = st.tabs(["1 · Method", "2 · Measured latency", "3 · Measured batch scaling", "4 · Simulated LLM load test",
                    "5 · Report", "6 · On a real stack"])
    with tabs[0]:
        _perf_method()
    with tabs[1]:
        _measured_latency()
    with tabs[2]:
        _measured_batching()
    with tabs[3]:
        _load_test()
    with tabs[4]:
        _report()
    with tabs[5]:
        _perf_real_stack()


def _perf_method() -> None:
    st.subheader("A fair performance test, step by step")
    for i, line in enumerate([
        "**Fix the workload**: model, precision, input/output lengths, request rate or concurrency, dataset.",
        "**Fix the environment**: hardware, driver, runtime, framework and server versions, settings. Write them down.",
        "**Warm up** before timing: the first calls include compilation, allocation and caching.",
        "**Measure distributions, not averages**: report P50, P95 and P99 with the sample count.",
        "**Sweep the load**: latency at one user tells you nothing about latency at 64.",
        "**Judge against the SLO**: report goodput, the share of requests that met every target.",
        "**Repeat** at least 3 runs and report the spread. Keep the raw results.",
    ], 1):
        st.markdown(f"{i}. {line}")
    st.warning("Never compare two numbers unless steps 1 and 2 match, and never infer real throughput from "
               "theoretical FLOPS.")


def _measured_latency() -> None:
    st.subheader("Measured: single-request latency of the real model")
    a, b, c = st.columns(3)
    n = a.select_slider("Timed requests", [100, 300, 1000, 3000], 300, key="mp_n")
    warm = b.select_slider("Warm-up requests (not counted)", [0, 5, 20, 50], 20, key="mp_warm")
    slo = c.number_input("P95 target (ms)", 0.001, 100.0, 0.10, 0.01, format="%.3f", key="mp_slo")
    st.button("Re-run measurement", key="mp_rerun")
    res = ml.measure_latency(ml.trained_model(), requests=n, warmup=warm)
    s = res["summary"]
    st.markdown(KIND_TEXT["measured"] + " Microsecond-scale numbers jump between runs and machines; that is the "
                "lesson, not a bug.")
    m = st.columns(4)
    m[0].metric("P50 (ms)", f"{s['p50']:.4f}")
    m[1].metric("P95 (ms)", f"{s['p95']:.4f}")
    m[2].metric("P99 (ms)", f"{s['p99']:.4f}")
    m[3].metric("First call (ms)", f"{res['first_call_ms']:.4f}" if res["first_call_ms"] is not None else "n/a")
    (st.success if s["p95"] <= slo else st.error)(
        f"P95 {s['p95']:.4f} ms {'meets' if s['p95'] <= slo else 'misses'} the {slo:g} ms target over {s['count']} requests.")
    fig = go.Figure(go.Histogram(x=res["samples_ms"], nbinsx=40, marker_color=ACCENT))
    for q, color in (("p50", GREY), ("p95", "#E67E22"), ("p99", RED)):
        fig.add_vline(x=s[q], line_color=color, annotation_text=q.upper())
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="latency (ms)", yaxis_title="requests")
    st.plotly_chart(fig, width="stretch")
    if res["warmup_ms"]:
        fig2 = go.Figure(go.Scatter(y=res["warmup_ms"] + res["samples_ms"][:80], mode="lines+markers",
                                    marker_color=ACCENT, line_width=1))
        fig2.add_vrect(x0=-0.5, x1=len(res["warmup_ms"]) - 0.5, fillcolor=GREY, opacity=0.2, line_width=0,
                       annotation_text="warm-up (discarded)")
        fig2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="request #",
                           yaxis_title="latency (ms)")
        st.plotly_chart(fig2, width="stretch")
    st.caption("The long right tail (P99 far above P50) comes from the OS, Python and caches, not the model. Real "
               "accelerators have tails too, which is why SLOs are written on P95/P99.")


def _measured_batching() -> None:
    st.subheader("Measured: batching trades latency for throughput")
    rows = ml.measure_batch_scaling(ml.trained_model())
    st.markdown(KIND_TEXT["measured"] + " Best of 5 runs per batch size.")
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_bar(x=[str(b) for b in df.batch_size], y=df.items_per_s, name="items/s", marker_color=ACCENT)
    fig.add_scatter(x=[str(b) for b in df.batch_size], y=df.batch_latency_ms, name="batch latency (ms)",
                    yaxis="y2", mode="lines+markers", marker_color=RED)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="batch size",
                      yaxis_title="throughput (items/s)", yaxis2=dict(title="latency (ms)", overlaying="y", side="right"),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df, hide_index=True, width="stretch")
    st.markdown("Throughput climbs with batch size while every request in the batch waits longer. Past some batch "
                "size the gain flattens. LLM servers make the same trade-off continuously; that's Part B.")


def _load_test() -> None:
    st.subheader("Simulated: replay a request trace against an LLM server")
    trace = ml.load_trace()
    st.markdown(f"Trace: **{len(trace)} fictional requests** over {trace[-1]['arrival_s']:.0f} s (about "
                f"{len(trace) / trace[-1]['arrival_s']:.1f} requests/s), with varying input and output lengths. "
                "The server is a toy continuous-batching model: every step admits queued requests up to the batch "
                "limit (paying their prefill) and generates one token for every running request.")
    st.markdown(KIND_TEXT["simulated"])
    a, b = st.columns(2)
    ttft_slo = a.number_input("TTFT SLO (ms)", 50, 20000, 2000, 100, key="mp_ttft_slo")
    tpot_slo = b.number_input("TPOT SLO (ms/token)", 5, 500, 50, 5, key="mp_tpot_slo")
    cols = st.columns(2)
    configs = []
    for i, (col, default_batch, default_rate) in enumerate(zip(cols, (8, 32), (1.0, 1.0))):
        with col.container(border=True):
            st.markdown(f"**Config {'AB'[i]}**")
            mb = st.select_slider("Max batch", [1, 2, 4, 8, 16, 32, 64, 128], default_batch, key=f"mp_mb{i}")
            rate = st.select_slider("Load (x trace rate)", [0.25, 0.5, 1.0, 1.5, 2.0, 4.0], default_rate, key=f"mp_rate{i}")
            res = ml.simulate_load_test(trace, rate_multiplier=rate, max_batch=mb)
            gp = ml.goodput(res["per_request"], ttft_slo, tpot_slo)
            configs.append((f"Config {'AB'[i]} (batch {mb}, {rate:g}x)", res, gp))
            m1, m2 = st.columns(2)
            m1.metric("TTFT P95 (ms)", f"{res['ttft_ms']['p95']:,.0f}")
            m2.metric("TPOT P50 (ms)", f"{res['tpot_ms']['p50']:.1f}")
            m3, m4 = st.columns(2)
            m3.metric("Output tokens/s", f"{res['output_tokens_per_s']:,.0f}")
            m4.metric("Goodput", f"{gp:.0%}")
    st.session_state["mp_last_load"] = [(name, {k: v for k, v in res.items() if k != "per_request"}, gp)
                                        for name, res, gp in configs]
    fig = go.Figure()
    for (name, res, _), color in zip(configs, (GREY, ACCENT)):
        fig.add_scatter(x=[r["request_id"] for r in res["per_request"]], y=[r["ttft_ms"] for r in res["per_request"]],
                        mode="markers", name=name, marker=dict(color=color, size=6))
    fig.add_hline(y=ttft_slo, line_dash="dash", line_color=RED, annotation_text="TTFT SLO")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="request (in arrival order)",
                      yaxis_title="TTFT (ms)", xaxis_showticklabels=False, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame([{"config": n, "TTFT P50": r["ttft_ms"]["p50"], "TTFT P95": r["ttft_ms"]["p95"],
                                "TPOT P50": r["tpot_ms"]["p50"], "TPOT P95": r["tpot_ms"]["p95"],
                                "E2E P95": r["e2e_ms"]["p95"], "tokens/s": r["output_tokens_per_s"],
                                "req/s": r["request_throughput_rps"], "goodput": f"{g:.0%}"}
                               for n, r, g in configs]).round(1), hide_index=True, width="stretch")
    st.markdown("**What to notice:** with a small batch limit, requests queue and TTFT explodes while TPOT looks "
                "great. Push the load above what the server can sustain and TTFT grows for every later request: "
                "the queue never drains. Throughput alone hides both, which is why POC exit criteria should use "
                "goodput at a stated load.")


def _report() -> None:
    st.subheader("Write it up: a benchmark report you can download")
    last = st.session_state.get("mp_last_load")
    lat = ml.measure_latency(ml.trained_model(), requests=300, warmup=20)["summary"]
    lines = ["# Benchmark report (training exercise)", "",
             "All customers and data are fictional. Part A is measured on the machine running the app; Part B is a "
             "toy simulation and is not a measurement of any product.", "",
             "## Environment", "- Model: tiny NumPy ticket classifier (" + f"{ml.trained_model().parameter_count():,}"
             + " parameters)", "- Hardware/software: <fill in: CPU, OS, Python and NumPy versions>", "",
             "## Part A: measured single-request latency (300 requests after 20 warm-up)",
             f"- P50 {lat['p50']:.4f} ms, P95 {lat['p95']:.4f} ms, P99 {lat['p99']:.4f} ms", ""]
    if last:
        lines += ["## Part B: simulated trace replay", "| config | TTFT P95 (ms) | TPOT P50 (ms) | tokens/s | goodput |",
                  "|---|---|---|---|---|"]
        lines += [f"| {n} | {r['ttft_ms']['p95']:,.0f} | {r['tpot_ms']['p50']:.1f} | {r['output_tokens_per_s']:,.0f} | {g:.0%} |"
                  for n, r, g in last]
        lines.append("")
    else:
        lines += ["## Part B", "Open the *Simulated LLM load test* tab first to include its results.", ""]
    lines += ["## Method notes", "- Same workload and environment for every compared run.",
              "- Percentiles reported with sample counts. Raw data kept.", "",
              "## Conclusion and next step", "<one sentence the customer can act on>"]
    report = "\n".join(lines)
    st.text_area("Report (Markdown)", report, height=360, key="mp_report_preview", disabled=True)
    st.download_button("Download report (.md)", report, file_name="benchmark_report.md", mime="text/markdown",
                       key="mp_report_dl")


def _perf_real_stack() -> None:
    st.subheader("The same steps on a real serving stack")
    st.markdown("A typical LLM serving benchmark replays a dataset at a fixed request rate against an "
                "OpenAI-compatible server and reports TTFT, TPOT, ITL and throughput percentiles.")
    _illustrative("# 1. Start the server (versions, flags and device options depend on your build)\n"
                  "vllm serve <your-model>\n\n"
                  "# 2. Replay a synthetic workload at a fixed rate and save raw results\n"
                  "vllm bench serve --model <your-model> --dataset-name random \\\n"
                  "  --random-input-len 1024 --random-output-len 128 --num-prompts 200 \\\n"
                  "  --request-rate 4 --save-result",
                  note="Flag names change between releases; run `vllm bench serve --help` on your version.")
    st.markdown("Then sweep the request rate (or concurrency), repeat each point at least 3 times, and plot "
                "latency percentiles against throughput, exactly like Part B.")
    render_sources(["vllm-bench-cli", "vllm-gpu-install"])
