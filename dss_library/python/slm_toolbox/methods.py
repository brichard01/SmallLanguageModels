"""The four inference techniques, each reduced to `run(description) -> dict`.

Every method returns the same shape so the flow can score them identically:

    {"method": str, "code": str|None, "steps": int, "reward": float, "detail": str}

The techniques themselves are task-agnostic — swap the episode runner's
environment and they apply unchanged to the next task bench.
"""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .config import get_llm
from .episode import run_episode, transcript
from . import inspect_env as ienv


# Traces are the evidence a reviewer reads, but a full self-consistency run is
# five transcripts long; cap what lands in the flow so the dataset stays readable.
DETAIL_MAX_CHARS = 6000


def _result(method, code, steps, reward, detail=""):
    if detail and len(detail) > DETAIL_MAX_CHARS:
        detail = detail[:DETAIL_MAX_CHARS] + "\n... [trace truncated]"
    return {
        "method": method,
        "code": code,
        "steps": int(steps or 0),
        "reward": float(reward or 0),
        "detail": detail,
    }


# --------------------------------------------------------------------------
# 1. Baseline — a single ReAct episode. The reference point for everything else.
# --------------------------------------------------------------------------
def baseline(description, max_steps=15, llm=None):
    ep = run_episode(description, llm=llm, max_steps=max_steps)
    return _result("baseline", ep["code"], ep["steps"], ep["reward"], transcript(ep))


# --------------------------------------------------------------------------
# 2. Self-consistency — n independent episodes, majority vote on the code.
#    Wang et al., 2023 (arXiv:2203.11171).
# --------------------------------------------------------------------------
def self_consistency(description, n=5, max_steps=15, temperature=1, llm=None):
    llm = llm or get_llm()
    with ThreadPoolExecutor(max_workers=n) as pool:
        runs = list(pool.map(
            lambda _: run_episode(description, llm=llm, max_steps=max_steps,
                                  temperature=temperature),
            range(n)))

    votes = Counter(r["code"] for r in runs if r["code"])
    winner = votes.most_common(1)[0][0] if votes else None
    chosen = next((r for r in runs if r["code"] == winner), runs[0])

    detail = f"votes={dict(votes)} over {n} episodes\n\n" + transcript(chosen)
    return _result("self_consistency", winner, sum(r["steps"] for r in runs),
                   chosen["reward"], detail)


# --------------------------------------------------------------------------
# 3. Self-Refine — generate, self-critique, re-run guided by the critique.
#    Madaan et al., 2023 (arXiv:2303.17651).
# --------------------------------------------------------------------------
FEEDBACK_PROMPT = """You are reviewing an attempt to classify a product into its correct 6-digit HS code.

Product description:
{description}

The attempt:
{transcript}

Critique this attempt. Be actionable and specific:
- Is the submitted 6-digit code the most accurate fit for the product?
- Point to concrete alternatives: a sibling heading/subheading or a different chapter
  that was skipped and might fit better, and say why.
- If the navigation stopped too early or forced the product into an ill-fitting code, say so.

Then decide whether the classification is already correct and needs no change.
Finish your answer with exactly one line:
STOP: yes    (if the submitted code is correct and no refinement is needed)
STOP: no     (if a different code should be explored)"""


def _feedback(description, attempt_text, llm):
    completion = llm.new_completion()
    completion.with_message(
        FEEDBACK_PROMPT.format(description=description, transcript=attempt_text),
        role="user")
    text = completion.execute().text or ""
    last = text.strip().splitlines()[-1].lower() if text.strip() else ""
    return text, ("stop:" in last and "yes" in last)


def self_refine(description, max_iters=3, max_steps=15, llm=None):
    llm = llm or get_llm()
    ep = run_episode(description, llm=llm, max_steps=max_steps)
    attempt = transcript(ep)
    history = [attempt]
    total_steps = ep["steps"]
    rounds = 0

    for _ in range(max_iters):
        feedback, should_stop = _feedback(description, attempt, llm)
        if should_stop:
            break
        rounds += 1
        guidance = (
            "A reviewer critiqued your previous classification attempt(s). "
            "Re-navigate the hierarchy from the start and act on this feedback; "
            "explore the alternatives it names before submitting.\n\n"
            "=== Previous attempts and feedback ===\n"
            + "\n\n".join(history)
            + f"\n\nReviewer feedback:\n{feedback}"
        )
        ep = run_episode(description, guidance=guidance, llm=llm, max_steps=max_steps)
        attempt = transcript(ep)
        total_steps += ep["steps"]
        history.append(f"Reviewer feedback:\n{feedback}\n\nResulting attempt:\n{attempt}")

    detail = f"refine_rounds={rounds}\n\n" + "\n\n".join(history)
    return _result("self_refine", ep["code"], total_steps, ep["reward"], detail)


# --------------------------------------------------------------------------
# 4. Submit-then-verify — provisional submission forces a re-inspection pass
#    over sibling branches before the answer is final.
# --------------------------------------------------------------------------
def inspect_submit(description, max_steps=30, llm=None):
    llm = llm or get_llm()
    env = ienv.InspectEnv()
    env.reset()

    completion = llm.new_completion()
    completion.settings["tools"] = ienv.TOOLS
    completion.with_message(ienv.SYSTEM_PROMPT, role="system")
    completion.with_message(description, role="user")

    trace = []
    for _ in range(max_steps):
        response = completion.execute()
        if not response.tool_calls:
            break
        completion.with_tool_calls(response.tool_calls)
        for tc in response.tool_calls:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"] or "{}")
            result = getattr(env, name)(**args)
            trace.append(f"{name}({args}) -> {result}")
            completion.with_tool_output(str(result), tool_call_id=tc["id"])
        if env.done:
            break

    detail = (f"Submitted code: {env.submitted or 'NONE'} (finished={env.done})\n\n"
              + "\n".join(trace))
    return _result("inspect_submit", env.submitted, env.total_calls, env.reward, detail)


METHODS = {
    "baseline": baseline,
    "self_consistency": self_consistency,
    "self_refine": self_refine,
    "inspect_submit": inspect_submit,
}


def run_method(name, description, **kwargs):
    """Entry point every method agent calls. Returns a JSON string for the flow."""
    try:
        result = METHODS[name](description, **kwargs)
        result["error"] = ""
    except Exception as exc:  # surfaced as a row value, never a silent empty cell
        result = _result(name, None, 0, 0)
        result["error"] = f"{type(exc).__name__}: {exc}"
    return json.dumps(result)
