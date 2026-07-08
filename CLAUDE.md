# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

This is an LLM agent benchmark and training project for **HS Code classification** — given a product description, an LLM must navigate the Harmonized System hierarchy (Section → 2-digit chapter → 4-digit heading → 6-digit subheading) using tool calls and submit the correct 6-digit code.

## Running inference

**Single episode (MLX, Apple Silicon):**
```bash
python run_hscode.py
```

**Full benchmark across Qwen3-4B/8B/14B MLX models:**
```bash
python benchmark.py
```
The benchmark checkpoints after every row — re-running resumes from where it left off.

**Parallel episodes via vLLM server (requires a model served at `localhost:8000`):**
```bash
python qwen_hscode_agent.py
```

**GRPO fine-tuning (requires CUDA + BitsAndBytes):**
```bash
python grpo_hscode.py
```

## Architecture

### Core environment: `hscode_env.py`
`HSCodeEnv` is the stateful tool-calling environment. It exposes three tools to the LLM:
- `search_section_children(section_letter)` — lists 2-digit chapters in a section
- `search_code_children(code)` — lists children of a 2- or 4-digit code
- `submit_final_code(code)` — locks in the 6-digit answer

**Key constraint**: codes are only valid to search/submit if they were previously returned by a tool call. The env enforces this — no hallucinating codes. It also tracks a `reward` score (positive for correct navigation/submission, negative per tool call) used as the GRPO training signal.

### Two inference backends

| File | Backend | Use case |
|---|---|---|
| `run_hscode.py` | `mlx_lm` (Apple Silicon) | Local dev/benchmark |
| `qwen_hscode_agent.py` | `qwen-agent` + vLLM server | Parallel/server inference |

Both implement `run_episode()` with the same signature and return dict (`submitted`, `reward`, `correct`, `steps`, `messages`). They share `HSCodeEnv` and `SYSTEM_PROMPT` from `hscode_env.py`.

`run_hscode.py` manually parses `<tool_call>...</tool_call>` XML tags from raw model output; `qwen_hscode_agent.py` delegates tool dispatch to the `qwen-agent` library.

### Data files (`data/`)
- `harmonized_system_by_parent.csv` — the HS code tree, indexed by parent group; used live during episodes
- `sections_prepared.csv` — section letters (A–U) to names; injected into the system prompt
- `benchmark_dataset.csv` — benchmark set with `product_description`, `answer` (6-digit), `hs_2`, `hs_4`, `section`, `trickiness`
- `benchmark_<model>.csv` — per-model benchmark results written by `benchmark.py`

### GRPO training: `grpo_hscode.py`
Uses TRL's `GRPOTrainer` with `HSCodeEnv` passed as `environment_factory`. The reward function reads `env.reward` directly after each generation. Requires a separate `data/HSCode_full.csv` (not the benchmark set). Targets CUDA with 4-bit QLoRA on `Qwen/Qwen3-4B`.
