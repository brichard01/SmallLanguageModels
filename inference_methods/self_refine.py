"""Self-Refine (Madaan et al., 2023, arXiv:2303.17651) on the HS code task.

The paper: a single model M generates an output, gives itself actionable/specific
FEEDBACK on that output (plus a stop indicator), then REFINEs the output given the
full history of prior outputs and feedback. Iterate until the feedback says stop
or a max is reached. No training, three prompts (generate / feedback / refine).

Here the "output" is a full agentic classification episode: the model navigates
the HS hierarchy with tools and submits a 6-digit code. Self-Refine wraps that:

  1. GENERATE  — run one agentic episode -> a submitted code + a navigation trace.
  2. FEEDBACK  — same model critiques that attempt (is the code the best fit? which
                 sibling/branch was missed?) and emits STOP: yes/no.
  3. REFINE    — run a fresh episode guided by all prior attempts + feedback, so the
                 model can land on a different code. Codes still come only from tools.

We report the last refined episode as the final answer.
"""

import json
from openai import OpenAI
from hscode_env import HSCodeEnv, SYSTEM_PROMPT, TOOLS

client = OpenAI()


def _episode(row, guidance=None, model="gpt-5-nano", temperature=1, max_steps=15):
    """Run one agentic classification episode. Returns (env, transcript_text)."""
    env = HSCodeEnv()
    env.reset(answer=row["answer"], hs_2=row["hs_2"], hs_4=row["hs_4"], section=row["section"])

    user = row["product_description"]
    if guidance:
        user += "\n\n" + guidance

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    trace = []  # human-readable record of what the model did, for the feedback step
    for _ in range(max_steps):
        msg = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=temperature
        ).choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            break
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = getattr(env, tc.function.name)(**args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            trace.append(f"{tc.function.name}({args}) -> {result}")
        if env.submitted:
            break

    transcript = (
        f"Submitted code: {env.submitted or 'NONE'}\n\n"
        "Navigation the model performed (tool calls and their results):\n"
        + "\n".join(trace)
    )
    return env, transcript


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


def _feedback(row, transcript, model="gpt-5-nano"):
    """FEEDBACK step: same model critiques the attempt, returns (text, should_stop)."""
    prompt = FEEDBACK_PROMPT.format(description=row["product_description"], transcript=transcript)
    text = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
    ).choices[0].message.content or ""

    last = text.strip().splitlines()[-1].lower() if text.strip() else ""
    should_stop = "stop:" in last and "yes" in last
    return text, should_stop


def run(row, model="gpt-5-nano", max_iters=3, max_steps=15):
    # GENERATE: initial episode (y0).
    env, transcript = _episode(row, model=model, max_steps=max_steps)
    total_steps = env.total_calls
    history = [transcript]

    # Iterate FEEDBACK -> REFINE, carrying the full history like the paper (Eqn. 4).
    for _ in range(max_iters):
        feedback, should_stop = _feedback(row, transcript, model=model)
        if should_stop:
            break

        guidance = (
            "A reviewer critiqued your previous classification attempt(s). "
            "Re-navigate the hierarchy from the start and act on this feedback; "
            "explore the alternatives it names before submitting.\n\n"
            "=== Previous attempts and feedback ===\n"
            + "\n\n".join(history)
            + f"\n\nReviewer feedback:\n{feedback}"
        )
        env, transcript = _episode(row, guidance=guidance, model=model, max_steps=max_steps)
        total_steps += env.total_calls
        history.append(f"Reviewer feedback:\n{feedback}\n\nResulting attempt:\n{transcript}")

    return {
        "submitted": env.submitted,
        "reward": env.reward,
        "correct": env.submitted == row["answer"],
        "steps": total_steps,
    }


if __name__ == "__main__":
    import pandas as pd

    row = pd.read_csv("data/benchmark_dataset.csv", dtype=str).to_dict("records")[0]
    print("Product:", row["product_description"])
    print("Answer:", row["answer"])
    print("Result:", run(row))
