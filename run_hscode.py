import json
import re
import mlx_lm
from hscode_env import HSCodeEnv, SYSTEM_PROMPT, TOOLS

model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-MLX-4bit")
thinking = True
max_tokens = 4096

if thinking:
    sampler = mlx_lm.sample_utils.make_sampler(
        temp=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    )
else:
    sampler = mlx_lm.sample_utils.make_sampler(
        temp=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
    )

def run_episode(product_description, answer=None, hs_2=None, hs_4=None, section=None, max_steps=15, max_tokens_per_step=max_tokens):
    env = HSCodeEnv()
    env.reset(answer=answer, hs_2=hs_2, hs_4=hs_4, section=section)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{product_description}\n\nUse the tools, one at a time, to navigate the code hierarchy, then submit your final answer."},
    ]

    for step in range(max_steps):
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=thinking,
        )

        print(f"\n[Step {step + 1}]")
        response = ""
        for chunk in mlx_lm.stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens_per_step, sampler=sampler):
            print(chunk.text, end="", flush=True)
            response += chunk.text
        print()

        tool_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)
        if not tool_match:
            break

        call = json.loads(tool_match.group(1))
        tool_name = call["name"]
        tool_args = call.get("arguments", {})
        call_id = f"call_{step}"

        after_think = re.split(r'</think>', response, maxsplit=1)[-1]
        content = re.split(r'<tool_call>', after_think, maxsplit=1)[0].strip()
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
            }],
        })

        tool_func = getattr(env, tool_name, None)
        result = tool_func(**tool_args) if tool_func else f"Unknown tool: {tool_name}"
        print(f"\n-> {tool_name}({tool_args})\n<- {result}")

        messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

        if tool_name == "submit_final_code" and env.submitted:
            print(f"{answer=}")
            print(f"{env.submitted=}")
            print(f"{env.reward=}")
            break

    return {
        "submitted": env.submitted,
        "reward": env.reward,
        "correct": env.submitted == answer if answer else None,
        "steps": step + 1,
        "messages": messages,
    }



if __name__ == "__main__":
    test = {
        "product_description": "Cotton T-shirt for men, single jersey knit",
        "answer": "610910",
        "hs_2": "61",
        "hs_4": "6109",
        "section": "K",
    }

    result = run_episode(
        product_description=test["product_description"],
        answer=test["answer"],
        hs_2=test["hs_2"],
        hs_4=test["hs_4"],
        section=test["section"],
    )

    result.pop("messages", None)
    print(f"\nResult: {result}")