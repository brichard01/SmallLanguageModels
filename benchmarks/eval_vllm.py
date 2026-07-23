"""Evaluate an arbitrary Hugging Face model on the HS-code task with vLLM.

This is the "run a whole model through the bench" harness (as opposed to
`generic_benchmark.py`, which scores one `inference_methods/` method over an API
backend). It serves the same agentic `HSCodeEnv` tool-calling loop that the rest
of the toolbox uses, but drives it with a local vLLM engine so any open-weights
checkpoint on the Hub can be benchmarked — **the model is a runtime argument**
(`--model <hf-id>`), so pointing the Google Cloud job at a new checkpoint needs
no image rebuild.

It is deliberately simpler than `training_methods/distill_generate.py`: no
top-k logprobs, no per-token bookkeeping. We only keep the **completions** — the
full conversation per episode plus the graded outcome (submitted code, reward,
correct) — and push them to the Hub as a dataset named after the model, so the
transcripts are inspectable and comparable across models later.

Reasoning handling (`--keep-reasoning` / default off): Qwen3-style models emit a
`<think>...</think>` block. When reasoning is *not* kept, those blocks are
stripped from each assistant turn — both in the saved completions and in the
context re-fed to the model on the next turn (this matches the usual multi-turn
convention of not carrying prior chain-of-thought forward). When kept, the
reasoning stays in the conversation verbatim.

LoRA adapters (e.g. the distill/GRPO outputs) are evaluated with `--lora <hf-id>`:
the base model and rank are read from the adapter config, the tokenizer/chat
template are taken from the adapter repo, and the adapter names the run.

Run (on a CUDA box / the google_gpu A100), from the repo root:
    python -m benchmarks.eval_vllm \
        --model Qwen/Qwen3-4B --limit 20 \
        --repo-id <user>/hscode-eval-qwen3-4b
    # a fine-tuned LoRA adapter:
    python -m benchmarks.eval_vllm \
        --lora <user>/qwen3-4b-hscode-distill --keep-reasoning
"""

import argparse
import json
import os
import re
import sys

# Path anchor that works in BOTH layouts: the local repo (this file lives in
# benchmarks/, so the repo root is its parent) and the flattened Docker image
# (everything is copied next to hscode_env.py). Detect which by looking for the
# env module beside the script; chdir so the "data/..." paths resolve either way.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE if os.path.exists(os.path.join(_HERE, "hscode_env.py")) else os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import pandas as pd

from hscode_env import HSCodeEnv, SYSTEM_PROMPT, TOOLS

# Qwen3 emits tool calls as:  <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def build_user_prompt(product_description: str) -> str:
    return f"""{product_description}

Use the tools to navigate the code hierarchy and then submit your final answer"""


def normalize_dataset(path: str) -> pd.DataFrame:
    """Load the task data, accepting both the full training schema and the small
    benchmark schema (answer/section -> hs_6/section_letter). Codes are read as
    strings so leading zeros (e.g. "0709") survive."""
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


def clean_assistant_content(text: str, keep_reasoning: bool) -> str:
    """Assistant 'content' for the saved conversation and the next-turn prompt:
    drop the <tool_call> block (it is re-rendered from the structured tool_calls
    field) and, unless reasoning is kept, drop the <think> block too."""
    text = TOOL_CALL_RE.sub("", text)
    if not keep_reasoning:
        text = THINK_RE.sub("", text)
    return text.strip()


def dispatch_tool(env: HSCodeEnv, name: str, args: dict) -> str:
    """Execute a model tool call against the env and return the tool result."""
    if name == "search_section_children":
        return env.search_section_children(str(args.get("section_letter", "")))
    if name == "search_code_children":
        return env.search_code_children(str(args.get("code", "")))
    if name == "submit_final_code":
        return env.submit_final_code(str(args.get("code", "")))
    return f"Unknown tool: {name}"


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
    }


def build_prompt(tokenizer, messages, enable_thinking):
    return tokenizer.apply_chat_template(
        messages, tools=TOOLS, add_generation_prompt=True, tokenize=False,
        enable_thinking=enable_thinking,
    )


def advance_episode(state, req, tokenizer, keep_reasoning, max_turns):
    """Append this turn's assistant message (+ any tool result) to the running
    conversation. Return True if the episode should take another turn."""
    output_ids = [int(t) for t in req.outputs[0].token_ids]
    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    tc = parse_tool_call(text)
    content = clean_assistant_content(text, keep_reasoning)

    if tc is None:
        state["messages"].append({"role": "assistant", "content": content})
        return False

    name, args = tc
    state["messages"].append({
        "role": "assistant",
        "content": content,
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}],
    })
    result = dispatch_tool(state["env"], name, args)
    state["messages"].append({"role": "tool", "content": result, "name": name})

    # count of assistant turns so far
    turns = sum(1 for m in state["messages"] if m["role"] == "assistant")
    return name != "submit_final_code" and turns < max_turns


def finalize_episode(state):
    """Turn a finished episode into one completion row."""
    env, row = state["env"], state["row"]
    return {
        "episode_id": state["ep_id"],
        "product_description": str(row["product_description"]),
        "answer": str(row["hs_6"]),
        "hs_2": str(row["hs_2"]),
        "hs_4": str(row["hs_4"]),
        "section": str(row["section_letter"]),
        # full transcript minus the constant system prompt, JSON-encoded so it
        # survives the columnar dataset format
        "messages": json.dumps(state["messages"][1:], ensure_ascii=False),
        "submitted": env.submitted,
        "reward": float(env.reward),
        "correct": bool(env.submitted == str(row["hs_6"])),
        "steps": int(env.total_calls),
    }


def run_pool(llm, tokenizer, sampling_params, rows, keep_reasoning, enable_thinking,
             max_turns, pool_size, max_model_len, lora_request=None):
    """Batched agentic rollout. Keeps up to `pool_size` episodes active and issues
    ONE `llm.generate([...])` per round over all active episodes' next-turn
    prompts, so vLLM continuous-batches them. Finished episodes are refilled from
    the remaining rows so the batch stays saturated. Mirrors the robustness of
    distill_generate: episodes that would overflow the context finish gracefully,
    and a dead generate call keeps whatever is already finished."""
    pending = list(enumerate(rows))
    active = []
    completions, n_correct, n_done = [], 0, 0
    margin = 64

    def emit(s):
        nonlocal n_done, n_correct
        rec = finalize_episode(s)
        completions.append(rec)
        n_done += 1
        n_correct += int(rec["correct"])
        turns = sum(1 for m in s["messages"] if m["role"] == "assistant")
        print(f"[{n_done}/{len(rows)}] ep{s['ep_id']} turns={turns} "
              f"submitted={rec['submitted']} gold={rec['answer']} "
              f"correct={rec['correct']} reward={rec['reward']:.1f}", flush=True)

    def refill():
        while len(active) < pool_size and pending:
            active.append(init_episode(*pending.pop(0)))

    refill()
    while active:
        prompts, batch = [], []
        for s in active:
            ptext = build_prompt(tokenizer, s["messages"], enable_thinking)
            plen = len(tokenizer(ptext, add_special_tokens=False)["input_ids"])
            if plen >= max_model_len - margin:
                emit(s)  # context exhausted -> finish as-is (no submission)
            else:
                prompts.append(ptext)
                batch.append(s)

        if not prompts:
            break

        try:
            outs = llm.generate(prompts, sampling_params, use_tqdm=False,
                                lora_request=lora_request)
        except Exception as e:
            print(f"generate failed ({e!r}); keeping {n_done} finished episodes, "
                  f"dropping {len(batch)} in-flight", flush=True)
            break

        still = []
        for s, req in zip(batch, outs):
            if advance_episode(s, req, tokenizer, keep_reasoning, max_turns):
                still.append(s)
            else:
                emit(s)
        active = still
        refill()

    return completions, n_correct


def default_repo_name(model: str) -> str:
    """Derive an HF dataset repo name from the model id, e.g.
    'Qwen/Qwen3-4B' -> 'hscode-eval-qwen3-4b'."""
    tag = model.rstrip("/").split("/")[-1].lower()
    tag = re.sub(r"[^a-z0-9._-]+", "-", tag)
    return f"hscode-eval-{tag}"


def main():
    ap = argparse.ArgumentParser(description="Evaluate a Hugging Face model on the HS-code task with vLLM.")
    ap.add_argument("--model", default=None,
                    help="Hugging Face model id (e.g. Qwen/Qwen3-4B). With --lora this is "
                         "the BASE model; if omitted it is read from the adapter's config.")
    ap.add_argument("--lora", default=None,
                    help="Hugging Face id (or local path) of a LoRA adapter to evaluate on "
                         "top of --model. The adapter's repo name identifies the run when "
                         "naming the completions dataset.")
    ap.add_argument("--max-lora-rank", type=int, default=None,
                    help="override the LoRA rank (default: read from the adapter config)")
    ap.add_argument("--data", default="data/benchmark_dataset.csv")
    ap.add_argument("--out", default=None, help="local save_to_disk dir (default: data/eval_<model>)")
    ap.add_argument("--repo-id", default=None,
                    help="HF dataset repo to push completions to (default: hscode-eval-<model> "
                         "under your namespace). Pass '' to skip pushing.")
    ap.add_argument("--private", action="store_true", help="push as a private HF dataset")
    ap.add_argument("--keep-reasoning", action="store_true",
                    help="keep <think>...</think> blocks in the saved conversation and "
                         "the re-fed context (default: strip them)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="disable the model's thinking mode in the chat template "
                         "(enable_thinking=False)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=2048, help="max new tokens per turn")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--quantization", default=None,
                    help="vLLM quantization (e.g. awq_marlin for AWQ checkpoints); "
                         "omit for full/bf16 weights")
    ap.add_argument("--pool-size", type=int, default=64,
                    help="max concurrent episodes per batched generate round")
    ap.add_argument("--limit", type=int, default=None, help="only first N rows (smoke test)")
    args = ap.parse_args()

    if not args.model and not args.lora:
        ap.error("provide --model (a full model) and/or --lora (an adapter)")

    import time
    # Heavy deps imported lazily so `--help` works without a GPU box.
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    from datasets import Dataset

    # Resolve LoRA: read the base model + rank from the adapter config (unless
    # overridden), and load the tokenizer from the adapter repo so its (possibly
    # fine-tuned) chat template is used. The adapter id names the run.
    base_model, lora_rank = args.model, args.max_lora_rank
    tokenizer_src = args.model
    lora_local_path = None
    if args.lora:
        # Download the adapter to a local dir (LoRARequest wants a path) and read
        # its base model + rank. Local paths pass through unchanged.
        if os.path.isdir(args.lora):
            lora_local_path = args.lora
        else:
            from huggingface_hub import snapshot_download
            lora_local_path = snapshot_download(args.lora)
        adapter_cfg = json.load(open(os.path.join(lora_local_path, "adapter_config.json")))
        base_model = args.model or adapter_cfg.get("base_model_name_or_path")
        lora_rank = args.max_lora_rank or int(adapter_cfg.get("r", 16))
        tokenizer_src = lora_local_path  # adapter repo ships tokenizer + chat_template
    eval_model = args.lora or args.model  # identity used for naming/summary

    df = normalize_dataset(args.data)
    if args.limit:
        df = df.head(args.limit)
    rows = df.to_dict("records")
    print(f"Model: {eval_model}" + (f"  (LoRA on base {base_model}, rank {lora_rank})" if args.lora else ""))
    print(f"Loaded {len(rows)} rows from {args.data}")
    print(f"keep_reasoning={args.keep_reasoning}  thinking={not args.no_thinking}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    llm_kwargs = dict(
        model=base_model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=False,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    lora_request = None
    if args.lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = lora_rank
        lora_request = LoRARequest("adapter", 1, lora_local_path)
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    t0 = time.time()
    completions, n_correct = run_pool(
        llm, tokenizer, sampling_params, rows,
        keep_reasoning=args.keep_reasoning,
        enable_thinking=not args.no_thinking,
        max_turns=args.max_turns,
        pool_size=args.pool_size,
        max_model_len=args.max_model_len,
        lora_request=lora_request,
    )
    elapsed = time.time() - t0

    n = max(len(rows), 1)
    accuracy = n_correct / n
    avg_reward = sum(c["reward"] for c in completions) / max(len(completions), 1)
    print(f"\n=== {eval_model} ===")
    print(f"Accuracy:   {n_correct}/{len(rows)} = {accuracy:.1%}")
    print(f"Avg reward: {avg_reward:.2f}")
    print(f"Wall-clock: {elapsed:.1f}s ({len(rows) / elapsed:.2f} episodes/s)")

    # Stamp per-row model + a run-level summary onto every completion so the
    # dataset is self-describing when compared across models later.
    summary = {"model": eval_model, "base_model": base_model, "lora": args.lora,
               "n": len(rows), "accuracy": accuracy, "avg_reward": avg_reward,
               "keep_reasoning": args.keep_reasoning, "thinking": not args.no_thinking}
    for c in completions:
        c["model"] = eval_model
    ds = Dataset.from_list(completions)

    out = args.out or f"data/eval_{default_repo_name(eval_model).replace('hscode-eval-', '')}"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ds.save_to_disk(out)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved completions to {out}")

    repo_id = args.repo_id
    if repo_id is None:
        repo_id = default_repo_name(eval_model)
    if repo_id:  # '' skips the push
        if "/" not in repo_id:
            from huggingface_hub import whoami
            repo_id = f"{whoami(token=os.environ.get('HF_TOKEN'))['name']}/{repo_id}"
        ds.push_to_hub(repo_id, private=args.private)
        print(f"Pushed completions to hub: {repo_id}")


if __name__ == "__main__":
    main()
