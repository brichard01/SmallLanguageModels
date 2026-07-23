"""Generate a knowledge-distillation dataset from a large teacher (Qwen3-32B-AWQ).

Step 1 of a distillation pipeline: have the teacher solve the HS-code task
*agentically* (running the real `HSCodeEnv` tool-calling loop) and record, for
every teacher-generated token, the **top-k logprobs** — the signal a smaller
student (Qwen3-4B) will later be trained on.

Why top-k logprobs?  A good KD loss on each assistant token is

    L = alpha * CE(student, teacher_sampled_token)
        + (1 - alpha) * T**2 * KL( p_teacher || q_student )

The CE term needs the teacher's sampled token; the KL ("dark knowledge") term
needs the teacher's distribution.  Storing the top-k (default 20) logprobs per
token carries both, is compact, and is exactly what vLLM/AWQ exposes (full-vocab
logits are unavailable through vLLM and would be gigabytes).  Qwen3-32B and
Qwen3-4B share the *same* tokenizer/vocab, so teacher token ids map 1:1 to the
student — no vocab remapping needed.

This produces an **off-policy** dataset (the student trains on the teacher's
trajectories).  The natural upgrade, if the student drifts, is on-policy GKD
(student generates, teacher scores, reverse-KL/JSD) — but that needs the teacher
live at train time, so it is a separate, later method.

Output: one row per assistant turn, each a self-contained training sequence
    { episode_id, turn, product_description, answer, hs_2, hs_4, section,
      prompt_ids, output_ids, output_text, topk_ids, topk_logprobs,
      reward, correct, submitted }
saved with `datasets` (save_to_disk, and optionally push_to_hub).

Run (on a CUDA box / the google_gpu A100), from the repo root:
    python -m training_methods.distill_generate \
        --data data/benchmark_dataset.csv --limit 20 \
        --out data/distill_qwen3_32b --repo-id <user>/hscode-distill-qwen3-32b
"""

import argparse
import json
import os
import re

import pandas as pd

from hscode_env import HSCodeEnv, SYSTEM_PROMPT, TOOLS

# The teacher's tool calls are parsed out of its generated text. Qwen3 emits:
#   <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def build_user_prompt(product_description: str) -> str:
    return f"""{product_description}

Use the tools to navigate the code hierarchy and then submit your final answer"""


def normalize_dataset(path: str) -> pd.DataFrame:
    """Load the task data, accepting both the full training schema and the
    small benchmark schema (answer/section -> hs_6/section_letter). Codes are
    read as strings so leading zeros (e.g. "0709") survive."""
    df = pd.read_csv(path, dtype=str)
    if "hs_6" not in df.columns and "answer" in df.columns:
        df["hs_6"] = df["answer"]
    if "section_letter" not in df.columns and "section" in df.columns:
        df["section_letter"] = df["section"]
    return df


def parse_tool_call(text: str):
    """Return (name, arguments_dict) from the first <tool_call> block, or None."""
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
        return obj["name"], obj.get("arguments", {}) or {}
    except (json.JSONDecodeError, KeyError):
        return None


def strip_tool_call(text: str) -> str:
    """Assistant 'content' for re-templating the next turn: the reasoning text
    with the <tool_call> block removed (the block is re-rendered from the
    structured tool_calls field, so keeping it here would double it)."""
    return TOOL_CALL_RE.sub("", text).strip()


def dispatch_tool(env: HSCodeEnv, name: str, args: dict) -> str:
    """Execute a teacher tool call against the env and return the tool result."""
    if name == "search_section_children":
        return env.search_section_children(str(args.get("section_letter", "")))
    if name == "search_code_children":
        return env.search_code_children(str(args.get("code", "")))
    if name == "submit_final_code":
        return env.submit_final_code(str(args.get("code", "")))
    return f"Unknown tool: {name}"


def extract_topk(step_logprobs, k: int):
    """vLLM returns, per generated token, a dict {token_id: Logprob(logprob,...)}.
    Return parallel (ids, logprobs) lists, highest logprob first, capped at k.
    (vLLM may include the sampled token beyond top-k; we keep the k most likely.)"""
    if not step_logprobs:
        return [], []
    items = sorted(step_logprobs.items(), key=lambda kv: kv[1].logprob, reverse=True)[:k]
    ids = [int(tid) for tid, _ in items]
    logprobs = [float(lp.logprob) for _, lp in items]
    return ids, logprobs


def init_episode(ep_id, row):
    """Start one agentic episode: fresh env + the initial system/user messages."""
    env = HSCodeEnv()
    env.reset(
        answer=str(row["hs_6"]),
        hs_2=str(row["hs_2"]),
        hs_4=str(row["hs_4"]),
        section=str(row["section_letter"]),
    )
    return {
        "ep_id": ep_id,
        "row": row,
        "env": env,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(str(row["product_description"]))},
        ],
        "turns": [],
    }


def build_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=True,
    )


def advance_episode(state, req, tokenizer, k, max_turns):
    """Record this turn's generation + top-k logprobs on `state`, execute the tool,
    and return True if the episode should take another turn (else it is finished)."""
    gen = req.outputs[0]
    prompt_ids = [int(t) for t in req.prompt_token_ids]  # exact tokenization vLLM used
    output_ids = [int(t) for t in gen.token_ids]

    topk_ids, topk_logprobs = [], []
    for step in (gen.logprobs or []):
        ids, lps = extract_topk(step, k)
        topk_ids.append(ids)
        topk_logprobs.append(lps)

    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    state["turns"].append({
        "turn": len(state["turns"]),
        "prompt_ids": prompt_ids,
        "output_ids": output_ids,
        "output_text": text,
        "topk_ids": topk_ids,
        "topk_logprobs": topk_logprobs,
    })

    tc = parse_tool_call(text)
    if tc is None:
        state["messages"].append({"role": "assistant", "content": text})
        return False

    name, args = tc
    state["messages"].append({
        "role": "assistant",
        "content": strip_tool_call(text),
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}],
    })
    result = dispatch_tool(state["env"], name, args)
    state["messages"].append({"role": "tool", "content": result, "name": name})

    return name != "submit_final_code" and len(state["turns"]) < max_turns


def finalize_episode(state):
    """Turn a finished episode into its per-turn training rows + episode meta."""
    env, row = state["env"], state["row"]
    meta = {
        "answer": str(row["hs_6"]),
        "hs_2": str(row["hs_2"]),
        "hs_4": str(row["hs_4"]),
        "section": str(row["section_letter"]),
        "product_description": str(row["product_description"]),
        "reward": float(env.reward),
        "submitted": env.submitted,
        "correct": bool(env.submitted == str(row["hs_6"])),
    }
    records = [{"episode_id": state["ep_id"], **t, **meta} for t in state["turns"]]
    return records, meta


def run_pool(llm, tokenizer, sampling_params, rows, k, max_turns, pool_size, max_model_len):
    """Batched agentic rollout. Keeps up to `pool_size` episodes active and issues
    ONE `llm.generate([...])` per round over all active episodes' next-turn prompts,
    so vLLM continuous-batches them instead of running one prompt at a time. Finished
    episodes are refilled from the remaining rows so the batch stays saturated.

    Robustness: an episode whose prompt would exceed the context window is finished
    gracefully (never sent to vLLM, which would otherwise reject it and kill the
    engine); and if a generate call dies anyway, whatever is finished is still kept."""
    pending = list(enumerate(rows))  # (ep_id, row), consumed as episodes finish
    active = []
    all_records, n_correct, n_done = [], 0, 0
    margin = 64  # leave room for at least a few output tokens

    def emit(s):
        nonlocal n_done, n_correct
        records, meta = finalize_episode(s)
        all_records.extend(records)
        n_done += 1
        n_correct += int(meta["correct"])
        print(f"[{n_done}/{len(rows)}] ep{s['ep_id']} turns={len(s['turns'])} "
              f"submitted={meta['submitted']} gold={meta['answer']} "
              f"correct={meta['correct']} reward={meta['reward']:.1f}", flush=True)

    def refill():
        while len(active) < pool_size and pending:
            active.append(init_episode(*pending.pop(0)))

    refill()
    while active:
        prompts, batch = [], []
        for s in active:
            ptext = build_prompt(tokenizer, s["messages"])
            plen = len(tokenizer(ptext, add_special_tokens=False)["input_ids"])
            if plen >= max_model_len - margin:
                emit(s)  # context exhausted -> finish as-is (no submission)
            else:
                prompts.append(ptext)
                batch.append(s)

        if not prompts:
            break

        try:
            outs = llm.generate(prompts, sampling_params, use_tqdm=False)
        except Exception as e:  # engine died -> keep what's finished, stop cleanly
            print(f"generate failed ({e!r}); keeping {n_done} finished episodes, "
                  f"dropping {len(batch)} in-flight", flush=True)
            break

        still = []
        for s, req in zip(batch, outs):
            if advance_episode(s, req, tokenizer, k, max_turns):
                still.append(s)
            else:
                emit(s)
        active = still
        refill()

    return all_records, n_correct


def main():
    ap = argparse.ArgumentParser(description="Generate a KD dataset with teacher top-k logprobs.")
    ap.add_argument("--data", default="data/benchmark_dataset.csv")
    ap.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    ap.add_argument("--out", default="data/distill_qwen3_32b", help="local save_to_disk dir")
    ap.add_argument("--repo-id", default=None, help="if set, push_to_hub to this dataset repo")
    ap.add_argument("--private", action="store_true", help="push as a private HF dataset")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=2048, help="max new tokens per turn")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--pool-size", type=int, default=64,
                    help="max concurrent episodes per batched generate round. vLLM "
                         "runs as many as KV cache allows and streams the rest; "
                         "lower --max-model-len to fit more concurrently.")
    ap.add_argument("--limit", type=int, default=None, help="only first N rows (smoke test)")
    args = ap.parse_args()

    import time
    # Heavy deps imported lazily so `--help` / import works without a GPU box.
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from datasets import Dataset

    df = normalize_dataset(args.data)
    if args.limit:
        df = df.head(args.limit)
    rows = df.to_dict("records")
    print(f"Loaded {len(rows)} rows from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        quantization="awq_marlin",
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=False,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        logprobs=args.topk,  # <- top-k logprobs per generated token
    )

    t0 = time.time()
    records, n_correct = run_pool(
        llm, tokenizer, sampling_params, rows, args.topk, args.max_turns,
        args.pool_size, args.max_model_len,
    )
    elapsed = time.time() - t0

    print(f"\nTeacher accuracy: {n_correct}/{len(rows)} = {n_correct / max(len(rows), 1):.1%}")
    print(f"Total assistant turns (rows): {len(records)}")
    print(f"Wall-clock: {elapsed:.1f}s  "
          f"({len(rows) / elapsed:.2f} episodes/s, {len(records) / elapsed:.2f} turns/s)")

    ds = Dataset.from_list(records)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ds.save_to_disk(args.out)
    print(f"Saved dataset to {args.out}")

    if args.repo_id:
        # push_to_hub picks up HF_TOKEN from the env automatically. If repo-id is
        # just a name (no namespace), resolve the token's namespace via whoami.
        repo_id = args.repo_id
        if "/" not in repo_id:
            from huggingface_hub import whoami
            repo_id = f"{whoami(token=os.environ.get('HF_TOKEN'))['name']}/{repo_id}"
        ds.push_to_hub(repo_id, private=args.private)
        print(f"Pushed to hub: {repo_id}")


if __name__ == "__main__":
    main()
