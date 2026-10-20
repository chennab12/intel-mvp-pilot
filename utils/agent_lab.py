"""A deterministic, offline agent for the Agentic AI Inference Lab.

The "LLM" here is SIMULATED: a rule-based planner that emits the same Thought / Action /
Final-answer steps a real tool-calling model would, so every run is reproducible and free.
The tools are real Python functions over a fictional environment (content/model_lab/agent_env.yaml)
and the app's own calculators and glossary. Token counts and latencies per step are SIMULATED
estimates. User text is only matched with regular expressions; it is never executed.
No Streamlit imports here so everything is unit-testable.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from utils import calculators as calc
from utils.content import load_all

ENV_PATH = Path(__file__).resolve().parent.parent / "content" / "model_lab" / "agent_env.yaml"

SYSTEM_PROMPT = (
    "You are a POC operations assistant. Use only the tools listed. Cite tool results. "
    "Never run shell commands. If a request is destructive or out of scope, refuse."
)

# Simulated serving characteristics of the planner model (used for the per-step cost estimate).
SIM_PREFILL_MS_PER_1K = 60.0
SIM_TPOT_MS = 25.0
TOOL_LATENCY_MS = {"lookup_system": 12, "check_compatibility": 8, "estimate_memory": 2,
                   "find_issues": 15, "glossary": 3, "cost_per_million": 1}

# Typical published shape of an 8B-class decoder (32 layers, 8 KV heads, head dim 128).
# Labeled as an assumption in the tool output; check the real model's config.json.
ASSUMED_SHAPES = {8: (32, 8, 128), 7: (32, 32, 128), 70: (80, 8, 128)}

BLOCK_PATTERNS = [
    r"\brm\s+-\w*[rf]", r"\bsudo\b", r"\bmkfs\b", r"\bshutdown\b", r"\breboot\b", r"\bdd\s+if=",
    r"\bdrop\s+table\b", r"\bformat\s+(the\s+)?(disk|drive)\b", r"curl[^|]*\|\s*(ba)?sh", r"\bwipe\b",
    r"\b(password|api[_ ]?key|credential|secret)s?\b",
]

AGENT_FAULTS = {
    "None (well-behaved model)": None,
    "Model hallucinates a tool that doesn't exist": "hallucinated_tool",
    "Model repeats the same call in a loop": "loop",
    "Model answers without using tools": "skip_tools",
    "Model sends bad tool arguments": "bad_args",
}


@lru_cache(maxsize=1)
def load_env() -> dict:
    with open(ENV_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------- tools ----------------

def tool_lookup_system(system_id: str) -> dict:
    systems = load_env()["systems"]
    if system_id not in systems:
        return {"error": f"unknown system '{system_id}'. Known: {', '.join(systems)}"}
    return {"system_id": system_id, **systems[system_id]}


def tool_check_compatibility(runtime: str, driver: str, os: str, framework: str) -> dict:
    matrix = load_env()["compat_matrix"]
    row = matrix.get(str(runtime))
    if row is None:
        return {"validated": False, "reason": f"runtime {runtime} is not in the demo compatibility matrix"}
    problems = []
    if driver not in row["driver"]:
        problems.append(f"driver {driver} not validated with runtime {runtime} (validated: {', '.join(row['driver'])})")
    if framework not in row["framework"]:
        problems.append(f"framework {framework} not validated (validated: {', '.join(row['framework'])})")
    if os not in row["os"]:
        problems.append(f"OS {os} not validated (validated: {', '.join(row['os'])})")
    return {"validated": not problems, "problems": problems}


def tool_estimate_memory(params_billion: float, precision: str, sequences: int, context_tokens: int,
                         device_memory_gb: float, overhead_pct: float = 10.0) -> dict:
    if precision not in calc.BYTES_PER_PARAM:
        return {"error": f"unknown precision {precision}"}
    if not (0 < params_billion <= 2000 and 0 < sequences <= 4096 and 0 < context_tokens <= 1_000_000):
        return {"error": "arguments out of range"}
    shape = ASSUMED_SHAPES.get(int(params_billion), (32, 8, 128))
    weights = calc.weight_memory_bytes(params_billion, precision)
    kv = calc.kv_cache_bytes(*shape, calc.BYTES_PER_PARAM[precision], context_tokens, sequences)
    budget = calc.memory_budget(weights, kv, overhead_pct, 0.0, device_memory_gb)
    return {"assumed_shape": f"{shape[0]} layers, {shape[1]} KV heads, head dim {shape[2]} (assumption)",
            **{k: round(v, 2) if isinstance(v, float) else v for k, v in budget.items()}}


def tool_find_issues(system_id: str | None = None, keyword: str | None = None) -> dict:
    issues = load_env()["issues"]
    hits = []
    for key, issue in issues.items():
        text = " ".join(str(v) for v in issue.values()).lower()
        on_system = system_id and (system_id in text or _issue_system(key) == system_id)
        on_keyword = keyword and keyword.lower() in text
        if on_system or on_keyword:
            hits.append({"id": key, **issue})
    return {"issues": hits}


def _issue_system(issue_id: str) -> str | None:
    # Which fictional system an issue belongs to (kept here, not in the yaml, so the tool has to join data).
    return {"POC-123": "poc-node-01", "POC-131": "poc-node-02", "EDGE-9": "edge-kiosk-07"}.get(issue_id)


def tool_glossary(term: str) -> dict:
    for entry in load_all()["glossary"]:
        if entry["term"].lower() == term.lower():
            return {"term": entry["term"], "definition": entry["definition"]}
    return {"error": f"'{term}' is not in the glossary"}


def tool_cost_per_million(dollars_per_hour: float, tokens_per_second: float) -> dict:
    if dollars_per_hour < 0 or tokens_per_second <= 0:
        return {"error": "need dollars_per_hour >= 0 and tokens_per_second > 0"}
    return {"usd_per_million_tokens": round(calc.cost_per_million_tokens(dollars_per_hour, tokens_per_second), 2)}


TOOLS = {
    "lookup_system": (tool_lookup_system, "Get the inventory record for a POC system id."),
    "check_compatibility": (tool_check_compatibility, "Check runtime/driver/OS/framework against the demo matrix."),
    "estimate_memory": (tool_estimate_memory, "Estimate weights + KV cache + overhead vs device memory."),
    "find_issues": (tool_find_issues, "Find open issues by system id or keyword."),
    "glossary": (tool_glossary, "Look up a term in the app glossary."),
    "cost_per_million": (tool_cost_per_million, "Compute $ per 1M tokens from $/hour and tokens/s."),
}


# ---------------- guardrails ----------------

def input_guard(question: str) -> str | None:
    """Return a reason if the request must be refused before any model call."""
    q = question.lower()
    for pat in BLOCK_PATTERNS:
        if re.search(pat, q):
            return f"matched blocked pattern `{pat}`"
    if len(question) > 2000:
        return "request longer than 2,000 characters"
    return None


def validate_call(tool: str, args: dict) -> str | None:
    """Allowlist + argument validation before a tool runs."""
    if tool not in TOOLS:
        return f"tool '{tool}' is not on the allowlist ({', '.join(TOOLS)})"
    fn = TOOLS[tool][0]
    params = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    defaults = len(fn.__defaults__ or ())
    required = params[:len(params) - defaults]
    unknown = [a for a in args if a not in params]
    missing = [p for p in required if p not in args]
    if unknown or missing:
        return f"bad arguments for {tool}: unknown={unknown}, missing={missing}"
    return None


# ---------------- the simulated LLM (rule-based planner) ----------------

SYSTEM_RE = re.compile(r"\b(poc-node-\d+|edge-kiosk-\d+)\b")
CUSTOMER_TO_SYSTEM = {"woodgrove": "poc-node-01", "contoso": "poc-node-02", "vanarsdel": "edge-kiosk-07"}


def understand(question: str) -> dict:
    """What the simulated model 'reads' from the question. Regex only; nothing is executed."""
    q = question.lower()
    systems = SYSTEM_RE.findall(q) or [s for c, s in CUSTOMER_TO_SYSTEM.items() if c in q]
    intents = []
    if re.search(r"status update|status report|draft .*status", q):
        intents.append("status")
    if re.search(r"validated|compatib|supported combination|support matrix", q):
        intents.append("compat")
    if re.search(r"\bfit\b|\bfits\b|memory|oom", q) and re.search(r"\d+\s*b\b", q):
        intents.append("memory")
    if re.search(r"block|owner|who owns|open issues|issues", q) and "status" not in intents:
        intents.append("issues")
    if re.search(r"cost|\$|per million", q):
        intents.append("cost")
    m = re.search(r"what (?:does|is) ([a-z0-9][a-z0-9 \-/]*?)(?: mean)?\??$", q) or re.search(r"define ([a-z0-9 \-/]+)", q)
    if m and not intents:
        intents.append("glossary")
    nums = {
        "params": _num(r"(\d+(?:\.\d+)?)\s*b\b", q),
        "precision": (re.search(r"\b(fp32|bf16|fp16|fp8|int8|int4)\b", q) or [None, None])[1],
        "sequences": _num(r"(\d+)\s*(?:sequences|seqs|users|concurrent)", q),
        "context": _num(r"(\d+)\s*(?:tokens|token context|context)", q),
        "dollars": _num(r"\$\s*(\d+(?:\.\d+)?)", q),
        "tps": _num(r"(\d+(?:\.\d+)?)\s*(?:tokens per second|tok/s|tokens/s)", q),
        "term": m.group(1).strip() if m else None,
    }
    return {"systems": list(dict.fromkeys(systems)), "intents": intents, "values": nums}


def _num(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def _plan_calls(u: dict, obs: dict) -> list[tuple[str, dict, str]]:
    """The full list of (tool, args, thought) a careful model would make, given what it has seen so far."""
    calls = []
    system = u["systems"][0] if u["systems"] else None
    v = u["values"]
    needs_system = {"compat", "memory", "status"} & set(u["intents"]) or ("issues" in u["intents"] and system and
                                                                          "compat" in u["intents"])
    if system and needs_system:
        calls.append(("lookup_system", {"system_id": system}, f"I need the inventory record for {system} first."))
    rec = obs.get("lookup_system")
    if "compat" in u["intents"] and rec and "error" not in rec:
        calls.append(("check_compatibility", {"runtime": rec["runtime"], "driver": rec["driver"], "os": rec["os"],
                                              "framework": rec["framework"]},
                      f"Compare runtime {rec['runtime']} / driver {rec['driver']} against the compatibility matrix."))
    if "memory" in u["intents"] and rec and "error" not in rec:
        calls.append(("estimate_memory", {"params_billion": v["params"] or 8, "precision": (v["precision"] or "bf16").upper(),
                                          "sequences": int(v["sequences"] or 1), "context_tokens": int(v["context"] or 4096),
                                          "device_memory_gb": rec["device_memory_gb"]},
                      f"Estimate memory and compare with the {rec['device_memory_gb']} GB on this system."))
    if "issues" in u["intents"] or "status" in u["intents"]:
        if system:
            calls.append(("find_issues", {"system_id": system}, f"Look for open issues on {system}."))
        else:
            kw = "soak" if "soak" in str(u.get("question", "")).lower() else None
            calls.append(("find_issues", {"keyword": kw or "block"}, "Search issues for what is blocking."))
    if "glossary" in u["intents"] and v["term"]:
        calls.append(("glossary", {"term": v["term"].upper() if len(v["term"]) <= 5 else v["term"]},
                      f"Look up '{v['term']}' in the glossary rather than answering from memory."))
    if "cost" in u["intents"] and v["dollars"] is not None and v["tps"]:
        calls.append(("cost_per_million", {"dollars_per_hour": v["dollars"], "tokens_per_second": v["tps"]},
                      "Use the cost tool for exact arithmetic."))
    return calls


def simulated_llm(question: str, steps: list[dict], fault: str | None = None) -> dict:
    """One 'LLM call': decide the next action, or produce the final answer from observations."""
    u = understand(question)
    u["question"] = question
    obs = {s["action"]: s["observation"] for s in steps if s.get("action") and s.get("observation") is not None}
    done = [(s["action"], s.get("args")) for s in steps if s.get("action")]
    if fault == "skip_tools":
        return {"thought": "I'm confident I know this already.", "final": "It should be fine; no issues expected."}
    if fault == "loop" and steps:
        last = steps[-1]
        return {"thought": "Let me check that again to be sure.", "action": last["action"], "args": last["args"]}
    if fault == "hallucinated_tool" and not steps:
        return {"thought": "I'll just log in and look.", "action": "ssh_exec", "args": {"cmd": "cat /etc/os-release"}}
    for tool, args, thought in _plan_calls(u, obs):
        if (tool, args) not in done:
            if fault == "bad_args" and not steps:
                args = {"id": args.get("system_id", "?")} if tool == "lookup_system" else {**args, "verbose": True}
            return {"thought": thought, "action": tool, "args": args}
    return {"thought": "I have what I need to answer.", "final": compose_answer(u, obs)}


def compose_answer(u: dict, obs: dict) -> str:
    parts = []
    system = u["systems"][0] if u["systems"] else None
    rec = obs.get("lookup_system")
    if rec and "error" in rec:
        return f"I couldn't find that system: {rec['error']}."
    if "status" in u["intents"] and rec:
        issues = obs.get("find_issues", {}).get("issues", [])
        lines = [f"**Status: {system}** ({rec['customer']})", f"- State: {rec['status']}",
                 f"- Stack: runtime {rec['runtime']}, driver {rec['driver']}, {rec['framework']}, {rec['os']}"]
        lines += [f"- Risk: {i['id']} {i['title']} (owner: {i['owner']}, {i['status']}; blocks {i['blocks']})" for i in issues]
        lines.append("- Ask: confirm a date for the root cause on the open issue(s) above.")
        return "\n".join(lines)
    comp = obs.get("check_compatibility")
    if comp is not None and rec:
        if comp["validated"]:
            parts.append(f"{system} is on a validated combination (runtime {rec['runtime']}, driver {rec['driver']}).")
        else:
            parts.append(f"{system} is not validated: " + "; ".join(comp.get("problems") or [comp.get("reason", "")]) + ".")
    mem = obs.get("estimate_memory")
    if mem and "error" not in mem:
        verdict = "fits" if mem["fits"] else "does not fit"
        parts.append(f"Estimated total {mem['total_gb']:.1f} GB (weights {mem['weights_gb']:.1f} GB + KV cache "
                     f"{mem['kv_gb']:.1f} GB + overhead {mem['overhead_gb']:.1f} GB) {verdict} in {mem['device_gb']:g} GB "
                     f"per device. Model shape is an assumption: {mem['assumed_shape']}.")
    if "issues" in u["intents"] and "find_issues" in obs:
        issues = obs["find_issues"]["issues"]
        if issues:
            parts.append("Open issues: " + "; ".join(f"{i['id']} {i['title']} (owner: {i['owner']}, blocks {i['blocks']})"
                                                     for i in issues) + ".")
        else:
            parts.append("No open issues found.")
    g = obs.get("glossary")
    if g:
        parts.append(f"{g['term']}: {g['definition']}" if "error" not in g else g["error"])
    cost = obs.get("cost_per_million")
    if cost and "error" not in cost:
        parts.append(f"About ${cost['usd_per_million_tokens']:.2f} per 1M tokens at 100% utilization (computed).")
    return " ".join(parts) or "I can't answer that with the tools I have. Try a system id, a glossary term, a memory or cost question."


# ---------------- the agent loop ----------------

def _sim_call_cost(prompt_text: str, output_text: str) -> dict:
    p = calc.rough_token_count(prompt_text)
    o = max(1, calc.rough_token_count(output_text))
    return {"prompt_tokens": p, "completion_tokens": o,
            "llm_ms": round(SIM_PREFILL_MS_PER_1K * p / 1000 + SIM_TPOT_MS * o, 1)}


def run_agent(question: str, fault: str | None = None, max_steps: int = 6) -> dict:
    """ReAct loop: guard -> (LLM decides -> validate -> tool -> observe)* -> final answer."""
    steps: list[dict] = []
    reason = input_guard(question)
    if reason:
        return {"question": question, "status": "blocked", "final": f"Request blocked by input guardrail: {reason}. "
                "This assistant never runs shell or destructive commands.", "steps": [], "totals": _totals([]),
                "tools_used": []}
    tool_desc = "\n".join(f"- {k}: {d}" for k, (_, d) in TOOLS.items())
    status = "max_steps"
    final = f"Stopped: hit the {max_steps}-step limit without an answer."
    for n in range(1, max_steps + 1):
        transcript = "\n".join(f"{s['thought']} {s.get('action')} {s.get('args')} {s.get('observation')}" for s in steps)
        decision = simulated_llm(question, steps, fault)
        cost = _sim_call_cost(f"{SYSTEM_PROMPT}\n{tool_desc}\n{question}\n{transcript}",
                              decision["thought"] + str(decision.get("action", "")) + str(decision.get("args", ""))
                              + decision.get("final", ""))
        step = {"n": n, "thought": decision["thought"], **cost, "tool_ms": 0}
        if "final" in decision:
            step.update(action=None, args=None, observation=None, guard="ok")
            steps.append(step)
            status, final = "answered", decision["final"]
            break
        tool, args = decision["action"], decision["args"]
        step.update(action=tool, args=args)
        problem = validate_call(tool, args)
        if problem is None and (tool, args) in [(s["action"], s["args"]) for s in steps]:
            problem = "repeated identical call (loop guard)"
        if problem:
            step.update(observation={"error": problem}, guard="blocked")
        else:
            step.update(observation=TOOLS[tool][0](**args), guard="ok", tool_ms=TOOL_LATENCY_MS.get(tool, 5))
        steps.append(step)
    used = [s["action"] for s in steps if s.get("action") and s["guard"] == "ok"]
    return {"question": question, "status": status, "final": final, "steps": steps, "totals": _totals(steps),
            "tools_used": used}


def _totals(steps: list[dict]) -> dict:
    return {"llm_calls": len(steps),
            "prompt_tokens": sum(s["prompt_tokens"] for s in steps),
            "completion_tokens": sum(s["completion_tokens"] for s in steps),
            "latency_ms": round(sum(s["llm_ms"] + s["tool_ms"] for s in steps), 1),
            "guard_blocks": sum(1 for s in steps if s.get("guard") == "blocked")}


def run_eval(fault: str | None = None, usd_per_million_tokens: float = 2.0) -> dict:
    """Run every eval task; a task passes when the answer contains every required phrase."""
    rows = []
    for task in load_env()["eval_tasks"]:
        run = run_agent(task["task"], fault=fault)
        answer = run["final"].lower()
        missing = [m for m in task["must_contain"] if m.lower() not in answer]
        expected = list(task["expected_tools"])
        tool_ok = sorted(set(run["tools_used"])) == sorted(expected)
        tok = run["totals"]["prompt_tokens"] + run["totals"]["completion_tokens"]
        rows.append({"id": task["id"], "task": task["task"], "passed": not missing, "missing": missing,
                     "tools_expected": expected, "tools_used": run["tools_used"], "tool_choice_ok": tool_ok,
                     "steps": run["totals"]["llm_calls"], "tokens": tok, "latency_ms": run["totals"]["latency_ms"],
                     "status": run["status"], "answer": run["final"]})
    n = len(rows)
    tokens = sum(r["tokens"] for r in rows)
    return {"rows": rows, "success_rate": sum(r["passed"] for r in rows) / n,
            "tool_accuracy": sum(r["tool_choice_ok"] for r in rows) / n,
            "avg_steps": sum(r["steps"] for r in rows) / n, "total_tokens": tokens,
            "est_cost_usd": round(tokens / 1e6 * usd_per_million_tokens, 6),
            "total_latency_ms": round(sum(r["latency_ms"] for r in rows), 1)}
