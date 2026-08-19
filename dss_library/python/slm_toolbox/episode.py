"""One agentic classification episode over the HS code environment.

This is the unit every inference method is built from: the model navigates the
HS hierarchy with tools and submits a 6-digit code. The environment forbids any
code a tool has not returned, so the model cannot hallucinate an answer.
"""

import json

from hscode_env import HSCodeEnv, SYSTEM_PROMPT, TOOLS

from .config import get_llm


def run_episode(description, guidance=None, llm=None, max_steps=15, temperature=None):
    """Run one episode. Returns a dict with the submitted code and a readable trace."""
    llm = llm or get_llm()
    env = HSCodeEnv()
    env.reset(hs_2=None, hs_4=None, answer=None)

    completion = llm.new_completion()
    completion.settings["tools"] = TOOLS
    if temperature is not None:
        completion.settings["temperature"] = temperature
    completion.with_message(SYSTEM_PROMPT, role="system")

    user = description if not guidance else f"{description}\n\n{guidance}"
    completion.with_message(user, role="user")

    trace = []
    final_text = ""
    for _ in range(max_steps):
        response = completion.execute()

        if not response.tool_calls:
            final_text = response.text or ""
            break

        completion.with_tool_calls(response.tool_calls)
        for tc in response.tool_calls:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"] or "{}")
            result = getattr(env, name)(**args)
            trace.append(f"{name}({args}) -> {result}")
            completion.with_tool_output(str(result), tool_call_id=tc["id"])

        if env.submitted:
            break

    return {
        "code": env.submitted,
        "steps": env.total_calls,
        "reward": env.reward,
        "trace": trace,
        "text": final_text,
    }


def transcript(episode):
    """Human-readable rendering of an episode, used as input to critique steps."""
    return (
        f"Submitted code: {episode['code'] or 'NONE'}\n\n"
        "Navigation the model performed (tool calls and their results):\n"
        + "\n".join(episode["trace"])
    )
