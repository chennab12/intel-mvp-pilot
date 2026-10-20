"""Agentic AI Inference Lab: watch a tool-calling agent run step by step, with guardrails and an eval suite.

The model deciding each step is SIMULATED (a deterministic rule-based planner), so runs are free,
offline and reproducible. Tools are real Python functions over fictional data. Nothing typed is executed.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.ui import ACCENT, KIND_TEXT, page_header, render_sources
from utils import agent_lab as al

GREY = "#9AA5B1"
SAMPLES = [t["task"] for t in al.load_env()["eval_tasks"]]


def render() -> None:
    page_header("🤖 Agentic AI Inference Lab",
                "An **agent** is an LLM in a loop: it reads the request, decides which tool to call, reads the "
                "result, and repeats until it can answer. Every loop iteration is another **inference call**, so "
                "agents multiply latency, tokens and cost, and add new ways to fail. Here you run a POC-operations "
                "agent over a fictional environment, inspect every step, break it on purpose, and evaluate it.")
    st.info("🧪 **Simulation.** The 'LLM' is a deterministic rule-based planner standing in for a real model; the "
            "tools are real Python functions over fictional systems, issues and a demo compatibility matrix. Token "
            "counts and latencies are simulated estimates.")
    tabs = st.tabs(["1 · How it works", "2 · Run the agent", "3 · Break it", "4 · Evaluate", "5 · Environment",
                    "6 · With a real model"])
    with tabs[0]:
        _how_it_works()
    with tabs[1]:
        _run_tab()
    with tabs[2]:
        _break_tab()
    with tabs[3]:
        _eval_tab()
    with tabs[4]:
        _env_tab()
    with tabs[5]:
        _real_model_tab()


def _how_it_works() -> None:
    st.subheader("Typical agentic inference steps")
    st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EAF3FB", color="#0068B5", fontname="Helvetica"];
  q [label="1. Request"]; g [label="2. Input guardrail"]; llm [label="3. LLM call:\\nthink + choose tool"];
  v [label="4. Validate call\\n(allowlist, args)"]; t [label="5. Run tool"]; o [label="6. Observation\\nadded to context"];
  a [label="7. Final answer"]; e [label="8. Evaluate\\n+ log tokens/latency"];
  q -> g -> llm -> v -> t -> o -> llm; llm -> a [label="enough info"]; a -> e;
  g -> a [label="blocked", style=dashed]; v -> o [label="rejected", style=dashed]; }""", width="stretch")
    st.dataframe(pd.DataFrame([
        {"Step": "Input guardrail", "Why a TPM cares": "Destructive or out-of-scope requests stop before any model call."},
        {"Step": "LLM call (per step)", "Why a TPM cares": "Each step re-sends the growing context: prompt tokens grow "
         "every step, so cost and TTFT grow too."},
        {"Step": "Validate call", "Why a TPM cares": "Models invent tools and arguments. An allowlist and schema check "
         "turn that into a recoverable error."},
        {"Step": "Tool run", "Why a TPM cares": "Tool latency adds to user latency; slow or flaky tools dominate agent "
         "performance more often than the model does."},
        {"Step": "Step limit", "Why a TPM cares": "A loop guard caps cost when the model gets stuck."},
        {"Step": "Evaluation", "Why a TPM cares": "Agents are judged on task success and tool choice over a fixed task "
         "set, not on one good demo."},
    ]), hide_index=True, width="stretch")


def _show_run(run: dict) -> None:
    t = run["totals"]
    m = st.columns(4)
    m[0].metric("Status", run["status"])
    m[1].metric("LLM calls", t["llm_calls"])
    m[2].metric("Tokens (prompt + output)", f"{t['prompt_tokens']:,} + {t['completion_tokens']:,}")
    m[3].metric("Simulated latency", f"{t['latency_ms'] / 1000:.2f} s")
    if run["status"] == "blocked":
        st.error(run["final"])
        return
    for s in run["steps"]:
        icon = "🛑" if s["guard"] == "blocked" else ("✅" if s["action"] is None else "🔧")
        title = f"{icon} Step {s['n']}: " + (f"call `{s['action']}`" if s["action"] else "final answer")
        with st.expander(title, expanded=s["guard"] == "blocked"):
            st.markdown(f"**Thought:** {s['thought']}")
            if s["action"]:
                st.markdown("**Action:**")
                st.code(json.dumps({"tool": s["action"], "arguments": s["args"]}, indent=2), language="json")
                st.markdown("**Guardrail:** " + ("passed" if s["guard"] == "ok" else "**rejected**"))
                st.markdown("**Observation:**")
                st.code(json.dumps(s["observation"], indent=2, default=str), language="json", wrap_lines=True)
            st.caption(f"Simulated: {s['prompt_tokens']} prompt tokens, {s['completion_tokens']} output tokens, "
                       f"LLM {s['llm_ms']:.0f} ms, tool {s['tool_ms']} ms")
    (st.success if run["status"] == "answered" else st.warning)("**Final answer**")
    st.markdown(run["final"])
    steps = run["steps"]
    if steps:
        fig = go.Figure()
        fig.add_bar(x=[f"step {s['n']}" for s in steps], y=[s["prompt_tokens"] for s in steps], name="prompt tokens",
                    marker_color=ACCENT)
        fig.add_bar(x=[f"step {s['n']}" for s in steps], y=[s["completion_tokens"] for s in steps], name="output tokens",
                    marker_color=GREY)
        fig.update_layout(barmode="stack", height=260, margin=dict(l=10, r=10, t=30, b=10),
                          title="Context grows every step (simulated tokens)", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")


def _run_tab() -> None:
    st.subheader("Ask the agent")
    st.caption("Your text is only matched against patterns to pick tools; it is never executed. Fictional data only.")
    sample = st.selectbox("Pick a sample request", SAMPLES, key="ag_sample")
    custom = st.text_input("…or type your own", "", max_chars=300, key="ag_custom",
                           placeholder="e.g. Are there open issues on poc-node-02?")
    question = custom.strip() or sample
    max_steps = st.slider("Max steps (loop guard)", 1, 10, 6, key="ag_max_steps")
    _show_run(al.run_agent(question, max_steps=max_steps))
    st.markdown(KIND_TEXT["simulated"])


def _break_tab() -> None:
    st.subheader("Inject a model failure and watch the guardrails")
    fault_name = st.selectbox("Failure", list(al.AGENT_FAULTS)[1:], key="ag_fault")
    question = st.selectbox("Request", SAMPLES, index=0, key="ag_fault_q")
    run = al.run_agent(question, fault=al.AGENT_FAULTS[fault_name], max_steps=5)
    clean = al.run_agent(question, max_steps=5)
    a, b = st.columns(2)
    a.metric("LLM calls vs healthy run", run["totals"]["llm_calls"], delta=run["totals"]["llm_calls"] -
             clean["totals"]["llm_calls"], delta_color="inverse")
    b.metric("Tokens vs healthy run", run["totals"]["prompt_tokens"] + run["totals"]["completion_tokens"],
             delta=(run["totals"]["prompt_tokens"] + run["totals"]["completion_tokens"]) -
             (clean["totals"]["prompt_tokens"] + clean["totals"]["completion_tokens"]), delta_color="inverse")
    _show_run(run)
    st.markdown({
        "hallucinated_tool": "The allowlist rejects `ssh_exec`; the error goes back to the model, which recovers. "
                             "The answer is still right, but you paid for an extra step.",
        "loop": "The loop guard rejects the repeated call and the step limit ends the run. Without them, this would "
                "burn tokens until a timeout.",
        "skip_tools": "No guardrail fires: the answer is confident and wrong. Only the **evaluation** (next tab) "
                      "catches this, which is why evals matter more than demos.",
        "bad_args": "Argument validation rejects the call before the tool runs; the model retries with valid "
                    "arguments.",
    }[al.AGENT_FAULTS[fault_name]])


def _eval_tab() -> None:
    st.subheader("Evaluate the agent on a fixed task set")
    st.markdown("Each task has phrases the answer must contain and the tools it should use. Run it for the healthy "
                "model and for each failure, the way you'd gate a new model version or a new serving backend.")
    a, b = st.columns(2)
    fault_name = a.selectbox("Model behaviour", list(al.AGENT_FAULTS), key="ag_eval_fault")
    price = b.number_input("Price, $ per 1M tokens (assumption)", 0.0, 100.0, 2.0, 0.5, key="ag_price")
    ev = al.run_eval(al.AGENT_FAULTS[fault_name], usd_per_million_tokens=price)
    m = st.columns(4)
    m[0].metric("Task success", f"{ev['success_rate']:.0%}")
    m[1].metric("Tool-choice accuracy", f"{ev['tool_accuracy']:.0%}")
    m[2].metric("Avg LLM calls / task", f"{ev['avg_steps']:.1f}")
    m[3].metric("Est. cost for the suite", f"${ev['est_cost_usd']:.4f}")
    st.dataframe(pd.DataFrame([{"id": r["id"], "task": r["task"], "pass": "✅" if r["passed"] else "❌",
                                "tools ok": "✅" if r["tool_choice_ok"] else "❌",
                                "tools used": ", ".join(r["tools_used"]) or "none",
                                "expected": ", ".join(r["tools_expected"]) or "none (refuse)",
                                "steps": r["steps"], "tokens": r["tokens"], "missing": ", ".join(r["missing"])}
                               for r in ev["rows"]]), hide_index=True, width="stretch")
    rows = []
    for name, f in al.AGENT_FAULTS.items():
        e = al.run_eval(f, usd_per_million_tokens=price)
        rows.append({"behaviour": name, "success": e["success_rate"], "tool accuracy": e["tool_accuracy"],
                     "tokens": e["total_tokens"]})
    fig = go.Figure()
    fig.add_bar(y=[r["behaviour"] for r in rows], x=[r["success"] for r in rows], orientation="h",
                name="task success", marker_color=ACCENT)
    fig.add_bar(y=[r["behaviour"] for r in rows], x=[r["tool accuracy"] for r in rows], orientation="h",
                name="tool-choice accuracy", marker_color=GREY)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickformat=".0%",
                      yaxis_autorange="reversed", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    st.caption("Recovering failures (hallucinated tool, bad arguments) keep success high but cost more tokens: "
               "track cost per task in the eval, not just pass rate.")
    st.markdown(KIND_TEXT["simulated"])


def _env_tab() -> None:
    env = al.load_env()
    st.subheader("The fictional environment the tools read")
    st.caption("Demo-stack version numbers are invented for the exercise. They are not Intel support statements; "
               "real support facts are in the Software Stack Explorer and the source registry.")
    st.markdown("**Systems**")
    st.dataframe(pd.DataFrame([{"system": k, **v} for k, v in env["systems"].items()]), hide_index=True, width="stretch")
    st.markdown("**Demo compatibility matrix** (runtime → validated companions)")
    st.dataframe(pd.DataFrame([{"runtime": k, **{c: ", ".join(v) for c, v in row.items()}}
                               for k, row in env["compat_matrix"].items()]), hide_index=True, width="stretch")
    st.markdown("**Open issues**")
    st.dataframe(pd.DataFrame([{"id": k, **v} for k, v in env["issues"].items()]), hide_index=True, width="stretch")
    st.markdown("**Tools on the allowlist**")
    st.dataframe(pd.DataFrame([{"tool": k, "what it does": d} for k, (_, d) in al.TOOLS.items()]), hide_index=True,
                 width="stretch")


def _real_model_tab() -> None:
    st.subheader("The same loop with a real model")
    st.markdown("Swap the simulated planner for any OpenAI-compatible endpoint that supports tool calling (for "
                "example a vLLM server on your own hardware). The loop, the allowlist, the step limit and the eval "
                "stay the same; only `simulated_llm` changes.")
    st.caption("**Illustrative, not run by this app.** Needs a server you control, with a model that supports tool "
               "calling; check your server's docs for how to enable it.")
    st.code('''from openai import OpenAI   # pip install openai; works with OpenAI-compatible servers

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed-for-local")
tools = [{"type": "function", "function": {
    "name": "lookup_system", "description": "Get the inventory record for a POC system id.",
    "parameters": {"type": "object", "properties": {"system_id": {"type": "string"}},
                   "required": ["system_id"]}}}]
messages = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Is poc-node-02 on a validated combination?"}]

for step in range(6):                                   # loop guard
    reply = client.chat.completions.create(model="<your-model>", messages=messages,
                                           tools=tools, temperature=0)
    msg = reply.choices[0].message
    if not msg.tool_calls:                              # final answer
        print(msg.content); break
    messages.append(msg)
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        problem = validate_call(call.function.name, args)   # allowlist + argument check
        result = {"error": problem} if problem else TOOLS[call.function.name][0](**args)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    print(step, reply.usage)                            # log tokens per step''', language="python", wrap_lines=True)
    st.markdown("**Performance questions to ask on a real POC:** how many LLM calls does a typical task take, what is "
                "TTFT at that context length, does the server reuse the shared prompt prefix between steps, and how "
                "much of end-to-end latency is tools rather than the model?")
    render_sources(["vllm-gpu-install", "vllm-bench-cli"])
