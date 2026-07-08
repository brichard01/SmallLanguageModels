"""Baseline method: run the HS Code env with gpt-5-nano via raw OpenAI tool calls."""

import json
from openai import OpenAI
from hscode_env import HSCodeEnv, SYSTEM_PROMPT, TOOLS

client = OpenAI()


def run(row, model="gpt-5-nano", temperature=0.3, max_steps=15):
    env = HSCodeEnv()
    env.reset(answer=row["answer"], hs_2=row["hs_2"], hs_4=row["hs_4"], section=row["section"])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["product_description"]},
    ]

    for _ in range(max_steps):
        msg = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            temperature=temperature
        ).choices[0].message

        messages.append(msg)
        if not msg.tool_calls:
            break
        for tc in msg.tool_calls:
            result = getattr(env, tc.function.name)(**json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        if env.submitted:
            break

    return {
        "submitted": env.submitted,
        "reward": env.reward,
        "correct": env.submitted == row["answer"],
        "steps": env.total_calls,
    }


if __name__ == "__main__":
    import pandas as pd

    row = pd.read_csv("data/benchmark_dataset.csv", dtype=str).to_dict("records")[0]
    print("Product:", row["product_description"])
    print("Answer:", row["answer"])

    result = run(row)
    print("Result:", result)
