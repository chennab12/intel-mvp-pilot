"""Performance Playground: memory, throughput/latency, benchmark comparison and cost calculators."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.ui import ACCENT, page_header
from utils import calculators as calc

TOOLS = [
    "Model weight memory",
    "KV-cache memory",
    "Total memory budget",
    "Throughput vs latency (simulated)",
    "Benchmark comparison",
    "Cost per million tokens",
]
KV_BYTES = {"FP32": 4, "BF16 / FP16": 2, "FP8 / INT8": 1}


def render() -> None:
    page_header("📈 Performance Playground",
                "Calculators with visible formulas, units, assumptions and limitations. They produce **educational "
                "estimates**. They never predict a real product's throughput, and nothing here infers speed from "
                "theoretical FLOPS.")
    tool = st.radio("Tool", TOOLS, horizontal=True, key="perf_tool")
    st.divider()
    {
        TOOLS[0]: weights_tool, TOOLS[1]: kv_tool, TOOLS[2]: budget_tool,
        TOOLS[3]: throughput_tool, TOOLS[4]: benchmark_tool, TOOLS[5]: cost_tool,
    }[tool]()


def _limits(*lines: str) -> None:
    with st.expander("Assumptions and limitations"):
        for line in lines:
            st.markdown(f"- {line}")


def weights_tool() -> None:
    st.subheader("Approximate model-weight memory")
    st.latex(r"\text{weight bytes} = \text{parameters} \times \text{bytes per parameter}")
    a, b = st.columns(2)
    params = a.number_input("Parameters (billions)", 0.1, 2000.0, 7.0, 0.5, key="w_params")
    prec = b.selectbox("Precision", list(calc.BYTES_PER_PARAM), index=1, key="w_prec")
    w = calc.weight_memory_bytes(params, prec)
    m1, m2, m3 = st.columns(3)
    m1.metric("Bytes per parameter", f"{calc.BYTES_PER_PARAM[prec]:g}")
    m2.metric("Weights (GB, decimal)", f"{w / calc.GB:,.2f}")
    m3.metric("Weights (GiB, binary)", f"{w / calc.GIB:,.2f}")
    rows = [{"precision": p, "GB": round(calc.weight_memory_bytes(params, p) / calc.GB, 2)} for p in calc.BYTES_PER_PARAM]
    fig = go.Figure(go.Bar(x=[r["precision"] for r in rows], y=[r["GB"] for r in rows], marker_color=ACCENT,
                           text=[f"{r['GB']:g}" for r in rows], textposition="outside"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="GB (weights only)")
    st.plotly_chart(fig, width="stretch")
    _limits("Weights only: KV cache, activations, runtime workspaces and fragmentation are extra.",
            "Quantized formats often store extra scale/zero-point data, so real files are slightly larger.",
            "1 GB = 10^9 bytes; 1 GiB = 2^30 bytes. Device specs usually use GB.",
            "Known check: 7B at BF16 = 14.00 GB = 13.04 GiB.")


def kv_tool() -> None:
    st.subheader("Simplified KV-cache memory")
    st.latex(r"\text{KV bytes} = 2 \times L \times H_{kv} \times d_{head} \times b \times T \times S")
    st.caption("L = layers, H_kv = key/value heads, d_head = head dimension, b = bytes per element, "
               "T = tokens per sequence (input + output), S = concurrent sequences. The 2 is one key and one value.")
    c1, c2, c3 = st.columns(3)
    layers = c1.number_input("Layers (L)", 1, 200, 32, key="kv_layers")
    kv_heads = c2.number_input("KV heads (H_kv)", 1, 256, 8, key="kv_heads")
    head_dim = c3.number_input("Head dimension (d_head)", 16, 1024, 128, 16, key="kv_dim")
    c4, c5, c6 = st.columns(3)
    prec = c4.selectbox("KV precision", list(KV_BYTES), index=1, key="kv_prec")
    tokens = c5.number_input("Tokens per sequence (T)", 1, 2_000_000, 8192, 512, key="kv_tokens")
    seqs = c6.number_input("Concurrent sequences (S)", 1, 4096, 16, key="kv_seqs")
    per_token = calc.kv_cache_bytes(layers, kv_heads, head_dim, KV_BYTES[prec], 1, 1)
    total = calc.kv_cache_bytes(layers, kv_heads, head_dim, KV_BYTES[prec], tokens, seqs)
    m1, m2, m3 = st.columns(3)
    m1.metric("Per token", f"{per_token / calc.KIB:,.1f} KiB")
    m2.metric("Total (GiB)", f"{total / calc.GIB:,.2f}")
    m3.metric("Total (GB)", f"{total / calc.GB:,.2f}")
    xs = [1, 2, 4, 8, 16, 32, 64, 128]
    fig = go.Figure(go.Scatter(x=xs, y=[calc.kv_cache_bytes(layers, kv_heads, head_dim, KV_BYTES[prec], tokens, s) / calc.GB
                                        for s in xs], mode="lines+markers", line_color=ACCENT))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="concurrent sequences",
                      yaxis_title="KV cache (GB)", xaxis_type="log")
    st.plotly_chart(fig, width="stretch")
    _limits("Architecture inputs are examples. Take real values from the model's published config (layers, "
            "number of key/value heads, head dimension).",
            "Assumes standard attention with every layer caching full-length K and V. Sliding-window attention, "
            "multi-latent attention, or KV compression change the result.",
            "Serving engines allocate KV in blocks/pages and may reserve a fixed pool, so actual use differs.",
            "Known check: 32 layers, 8 KV heads, head dim 128, 2 bytes = 128 KiB per token; 8,192 tokens x 16 sequences = 16 GiB.")


def budget_tool() -> None:
    st.subheader("Total memory budget")
    st.latex(r"\text{total} = (W + KV)\times(1+\tfrac{o}{100}) + F")
    st.caption("W = weights, KV = KV cache, o = proportional overhead %, F = fixed overhead (GB).")
    c1, c2, c3 = st.columns(3)
    params = c1.number_input("Parameters (billions)", 0.1, 2000.0, 8.0, 0.5, key="b_params")
    prec = c2.selectbox("Weight precision", list(calc.BYTES_PER_PARAM), index=1, key="b_prec")
    device = c3.number_input("Device memory (GB)", 1.0, 1024.0, 32.0, 1.0, key="b_dev")
    c4, c5, c6 = st.columns(3)
    layers = c4.number_input("Layers", 1, 200, 32, key="b_layers")
    kv_heads = c5.number_input("KV heads", 1, 256, 8, key="b_kvh")
    head_dim = c6.number_input("Head dim", 16, 1024, 128, 16, key="b_hd")
    c7, c8, c9, c10 = st.columns(4)
    kv_prec = c7.selectbox("KV precision", list(KV_BYTES), index=1, key="b_kvp")
    tokens = c8.number_input("Tokens / sequence", 1, 2_000_000, 4096, 512, key="b_tok")
    seqs = c9.number_input("Concurrent sequences", 1, 4096, 8, key="b_seq")
    overhead = c10.number_input("Overhead %", 0.0, 100.0, 10.0, 1.0, key="b_ovh")
    fixed = st.number_input("Fixed overhead (GB): runtime, context, workspace", 0.0, 64.0, 1.0, 0.5, key="b_fix")
    w = calc.weight_memory_bytes(params, prec)
    kv = calc.kv_cache_bytes(layers, kv_heads, head_dim, KV_BYTES[kv_prec], tokens, seqs)
    res = calc.memory_budget(w, kv, overhead, fixed, device)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Weights", f"{res['weights_gb']:.2f} GB")
    m2.metric("KV cache", f"{res['kv_gb']:.2f} GB")
    m3.metric("Overhead", f"{res['overhead_gb']:.2f} GB")
    m4.metric("Total", f"{res['total_gb']:.2f} GB", delta=f"{res['headroom_gb']:+.2f} GB headroom",
              delta_color="normal" if res["fits"] else "inverse")
    if res["fits"]:
        st.success(f"Fits in {device:g} GB with {res['headroom_gb']:.2f} GB headroom (estimate).")
    else:
        st.error(f"Does not fit in one {device:g} GB device (estimate). Roughly {res['devices_needed']} devices' worth of "
                 "memory, if the model can be split; or reduce context, concurrency or precision.")
    per_seq = calc.kv_cache_bytes(layers, kv_heads, head_dim, KV_BYTES[kv_prec], tokens, 1)
    st.caption(f"At this context length, about {calc.max_sequences(device, w, overhead, fixed, per_seq)} full-length "
               "sequences fit on one device after weights and overhead (same overhead model).")
    fig = go.Figure(go.Bar(x=["Weights", "KV cache", "Overhead", "Headroom" if res["fits"] else "Shortfall"],
                           y=[res["weights_gb"], res["kv_gb"], res["overhead_gb"], abs(res["headroom_gb"])],
                           marker_color=[ACCENT, "#5B9BD5", "#A5A5A5", "#70AD47" if res["fits"] else "#C00000"]))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="GB (estimate)")
    st.plotly_chart(fig, width="stretch")
    _limits("Overhead varies widely by framework, serving engine, graph compilation and settings. Replace the "
            "default with a value measured on the real stack.",
            "Splitting a model across devices adds communication and per-device overhead; 'devices needed' is "
            "memory arithmetic only.",
            "Use vendor spec memory capacity, then subtract anything reserved by the driver/runtime if documented.")


def throughput_tool() -> None:
    st.subheader("Throughput vs latency (clearly simulated)")
    st.warning("**Simulated data.** A toy memory-bound decode model for teaching the *shape* of the tradeoff. "
               "It does not represent any Intel or other product.")
    st.latex(r"t_{step} = t_{weights} + k \times \min(C, B_{max}),\quad"
             r"\text{per-user tok/s} = \frac{1000}{t_{step}},\quad \text{aggregate} = \text{per-user}\times\min(C,B_{max})")
    c1, c2, c3 = st.columns(3)
    t_w = c1.slider("Weight-read time per step (ms)", 5.0, 80.0, 20.0, 1.0, key="t_w")
    k = c2.slider("Extra ms per sequence in batch (k)", 0.1, 3.0, 0.6, 0.1, key="t_k")
    bmax = c3.slider("Max batch (B_max)", 4, 256, 64, 4, key="t_b")
    c4, c5 = st.columns(2)
    inp = c4.slider("Input tokens", 100, 16000, 1000, 100, key="t_in")
    out = c5.slider("Output tokens", 10, 2000, 200, 10, key="t_out")
    levels = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]
    rows = calc.simulate_serving(levels, weight_read_ms=t_w, per_seq_ms=k, input_tokens=inp, output_tokens=out,
                                 max_batch=bmax)
    df = pd.DataFrame(rows)
    target = st.number_input("Per-user target (tokens/s)", 1.0, 200.0, 20.0, 1.0, key="t_target")
    ok = df[df["per_user_tokens_per_s"] >= target]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["concurrency"], y=df["aggregate_tokens_per_s"], name="aggregate tok/s",
                             line_color=ACCENT, mode="lines+markers"))
    fig.add_trace(go.Scatter(x=df["concurrency"], y=df["per_user_tokens_per_s"], name="per-user tok/s",
                             yaxis="y2", mode="lines+markers", line_dash="dot"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="concurrent requests",
                      yaxis=dict(title="aggregate tokens/s (simulated)"),
                      yaxis2=dict(title="per-user tokens/s (simulated)", overlaying="y", side="right"),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    if len(ok):
        best = ok.iloc[-1]
        st.success(f"Highest simulated concurrency meeting {target:g} tok/s per user: **{int(best['concurrency'])}** "
                   f"→ aggregate {best['aggregate_tokens_per_s']:,.0f} tok/s, TTFT P95 {best['ttft_p95_ms']:,.0f} ms.")
    else:
        st.error("No simulated operating point meets that per-user target.")
    st.dataframe(df, hide_index=True, width="stretch")
    _limits("Real decode time depends on kernels, memory bandwidth achieved, attention cost growing with context, "
            "scheduling and host overhead.",
            "Queueing above max batch is approximated, and TTFT jitter is random noise (seeded).",
            "Use this to learn which numbers to ask for, then measure the real system with a benchmark tool.")


DEFAULT_RUNS = [
    {"run": "System A run 1", "data_label": "simulated", "hardware": "Fictional system A", "software": "stack v1.0 (fictional)",
     "model": "open-8b rev1", "precision": "BF16", "input_tokens": 2000, "output_tokens": 256, "concurrency": 32,
     "warmup_excluded": "yes", "accuracy_gate": "pass", "p95_ttft_ms": 740, "output_tok_s": 960},
    {"run": "System B run 1", "data_label": "simulated", "hardware": "Fictional system B", "software": "stack v2.3 (fictional)",
     "model": "open-8b rev1", "precision": "INT8", "input_tokens": 2000, "output_tokens": 256, "concurrency": 64,
     "warmup_excluded": "no", "accuracy_gate": "not run", "p95_ttft_ms": 1210, "output_tok_s": 1450},
]


def benchmark_tool() -> None:
    st.subheader("Benchmark comparison with workload and environment metadata")
    st.caption("Edit the table (rows are fictional and labeled *simulated*). The checker flags anything that makes "
               "the comparison unfair before you look at the numbers.")
    df = st.data_editor(pd.DataFrame(DEFAULT_RUNS), num_rows="dynamic", key="bench_editor", width="stretch",
                        column_config={"data_label": st.column_config.SelectboxColumn(
                            "data_label", options=["measured", "vendor claim", "estimate", "simulated"])})
    runs = [r for r in df.to_dict("records") if str(r.get("run") or "").strip()]
    issues = calc.fairness_issues(runs)
    if issues:
        st.error("**Not a fair comparison yet:**\n\n" + "\n".join(f"- {i}" for i in issues))
    elif len(runs) >= 2:
        st.success("Settings match across runs. Differences in results can be attributed to the systems (within run-to-run variance).")
    if runs:
        fig = go.Figure([
            go.Bar(name="output tok/s", x=[r["run"] for r in runs], y=[r.get("output_tok_s") or 0 for r in runs],
                   marker_color=ACCENT),
            go.Bar(name="P95 TTFT (ms)", x=[r["run"] for r in runs], y=[r.get("p95_ttft_ms") or 0 for r in runs],
                   marker_color="#A5A5A5"),
        ])
        fig.update_layout(barmode="group", height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Chart labels follow each row's data_label. Never present simulated rows as measured results.")
    st.download_button("Download comparison (CSV)", df.to_csv(index=False), file_name="benchmark_comparison.csv",
                       mime="text/csv")
    _limits("Matching settings is necessary but not sufficient: also compare run-to-run variance and repeat counts.",
            "Record full inventories (firmware, driver, runtime, framework, serving engine) for every row.",
            "Vendor claims belong in their own rows labeled 'vendor claim' with the published configuration.")


def cost_tool() -> None:
    st.subheader("Illustrative cost per million tokens")
    st.latex(r"\text{cost per 1M tokens} = \frac{\text{hourly cost}}{\text{tokens/s}\times 3600\times u}\times 10^6")
    st.caption("u = utilization (share of time the system serves useful traffic). All inputs are yours; this app "
               "contains no real prices.")
    c1, c2, c3 = st.columns(3)
    hourly = c1.number_input("All-in hourly cost ($)", 0.0, 10000.0, 10.0, 0.5, key="c_hour")
    tps = c2.number_input("Throughput at the latency target (tokens/s)", 1.0, 1e7, 1000.0, 50.0, key="c_tps")
    util = c3.slider("Utilization (%)", 1, 100, 100, key="c_util")
    cost = calc.cost_per_million_tokens(hourly, tps, util)
    st.metric("Cost per 1M tokens", f"${cost:,.2f}")
    t1, t2 = st.columns(2)
    per_task = t1.number_input("Tokens per task (for example one support answer)", 1, 1_000_000, 2200, 100, key="c_task")
    t2.metric("Cost per task", f"${cost * per_task / 1e6:,.4f}")
    st.markdown("**Energy per token**")
    st.latex(r"\text{J/token} = \frac{\text{average power (W)}}{\text{tokens/s}}")
    e1, e2 = st.columns(2)
    watts = e1.number_input("Average measured power (W)", 1.0, 100000.0, 400.0, 10.0, key="c_w")
    e2.metric("Energy per token", f"{calc.energy_per_token(watts, tps):.4f} J")
    utils = [10, 25, 50, 75, 100]
    fig = go.Figure(go.Bar(x=[f"{u}%" for u in utils], y=[calc.cost_per_million_tokens(hourly, tps, u) for u in utils],
                           marker_color=ACCENT))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="utilization",
                      yaxis_title="$ per 1M tokens (illustrative)")
    st.plotly_chart(fig, width="stretch")
    _limits("Hourly cost should include hardware amortization or cloud rate, power, cooling, space, software and support.",
            "Use throughput measured at the customer's latency target, not peak throughput.",
            "State whether power is device-only, server or rack level.",
            "Known check: $10/h at 1,000 tok/s and 100% utilization = $2.78 per 1M tokens.")
