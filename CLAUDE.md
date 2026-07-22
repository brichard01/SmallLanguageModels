# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project objective (read this first)

The real goal of this project is **not** to solve HS code classification. It is to **build and document a reusable toolbox of methods for making Small Language Models (SLMs) perform well on targeted, narrow tasks.**

The thesis: a small model (e.g. Qwen3-4B) that is cheap and runs locally can, with the right scaffolding — prompting, tool design, decoding strategies, agentic loops, verification passes, fine-tuning, etc. — approach or match a large frontier model on a *specific* task. We want to find out **which techniques buy how much performance, at what cost**, and to be able to **apply the winning combination to a new task quickly.**

HS code classification is simply the **first test bench**. It is a good one: it is a well-scoped, hierarchical, tool-navigation task with a clear correctness signal (the 6-digit answer) and a graded reward. But the point of every method we write is that it should generalize to the *next* task we throw at it.

### Consequences of this objective — how to work here

1. **A method that performs badly on HS codes is NOT a failure and must NOT be deleted.** It stays in the toolbox. We may have implemented it imperfectly, or it may simply not suit this particular task but shine on the next one. What matters is that we *tried it and recorded what happened.*

2. **Every experiment must be documented in `EXPERIMENTS.md`** (the running experiment report — see below). The value of this repo compounds through that log: when we move to a new task, we replay the toolbox in roughly the order the log suggests, and we already know the trade-offs.

3. **Methods should be written to generalize.** The task-specific bits (the environment, the tools, the dataset) are separable from the *technique* (self-consistency, verification, best-of-n, tree search, fine-tuning…). When you add a method, keep the technique cleanly reusable so it can be repointed at a new task's environment with minimal change.

4. **We favor breadth then depth.** First get a broad, cheap read on many techniques (which are worth pursuing), then invest in tuning the promising ones.

## Repository layout

| Path | Role |
|---|---|
| `hscode_env.py` | The HS code task: `HSCodeEnv` (stateful tool-calling env + reward), `SYSTEM_PROMPT`, `TOOLS`. The "task bench." |
| `inference_methods/` | **The inference toolbox.** One file per prompting/decoding/agentic technique. Each exposes `run(row) -> dict`. |
| `training_methods/` | **The training toolbox.** Fine-tuning-based techniques (GRPO, etc.) that train on the env's reward. |
| `papers/` | Research papers (PDFs / notes) to read and turn into methods. **Source of new toolbox entries.** |
| `generic_benchmark.py` | Task-agnostic harness: `benchmark(fn)` scores any method's `run` over the dataset (accuracy + avg reward). |
| `benchmark.py` | Sweeps the *local MLX baseline* across Qwen3-4B/8B/14B. Checkpoints per row; resumes. |
| `run_hscode.py` | MLX (Apple Silicon) baseline backend — `run_episode()`, manual `<tool_call>` XML parsing. |
| `training_methods/grpo_hscode.py` | GRPO/QLoRA fine-tuning of Qwen3-4B on the env's reward (needs CUDA). A training-based method. |
| `google_gpu/` | Google Cloud Batch recipe for running GPU jobs (A100) — infra for the training-based methods. See its own README. |
| `data/` | Datasets + per-model benchmark result CSVs. |
| `EXPERIMENTS.md` | **The experiment report. Append to it after every experiment.** |

## The toolbox: `inference_methods/` and `training_methods/`

Inference-time techniques (prompting, decoding, agentic loops, verification) live
in `inference_methods/`; training-based techniques (fine-tuning on the env
reward) live in `training_methods/`. Each inference method is one file exposing a
single entry point with a stable signature:

```python
def run(row: dict) -> dict:
    # row has: product_description, answer, hs_2, hs_4, section, trickiness
    return {"submitted": ..., "reward": ..., "correct": bool, "steps": int}
```

Any method satisfying this contract is scorable by `generic_benchmark.py` with no changes to the harness. To benchmark a method, point the import in `generic_benchmark.py` at it and run `python generic_benchmark.py`.

Methods present today (each is a technique, not just an HS-code hack):

- `inference_methods/raw_openai.py` — **baseline**: single agent, raw OpenAI tool-calling loop over `HSCodeEnv`. The reference point every other method is measured against.
- `inference_methods/self_consistency.py` — **self-consistency**: run the baseline *n* times at high temperature, majority-vote the submitted code.
- `inference_methods/inspect_submit.py` — **submit-then-verify**: a rebuilt env where `submit_final_code` is provisional and triggers a mandatory re-inspection pass over sibling branches before a terminal `finish`. Explores "give the model a chance to self-correct."
- `inference_methods/self_refine.py` — **Self-Refine**: one agentic episode produces a code, then the same model critiques and refines it in a feedback→refine loop.
- `training_methods/grpo_hscode.py` — **GRPO/QLoRA fine-tuning**: trains Qwen3-4B on the env's graded reward (needs CUDA; run via `google_gpu/`).

When you add a method, keep the *technique* separable from the HS-code specifics so it can be repointed at a future task.

## `papers/` → `inference_methods/` / `training_methods/` workflow

`papers/` holds research papers we want to try. The loop is:

1. Read a paper in `papers/`.
2. Implement the **simplest faithful version** of its core idea as a new file in `inference_methods/` (respecting the `run(row)` contract) or `training_methods/` if it's training-based. Simple first — a rough but honest implementation that we can benchmark beats a perfect one we never finish.
3. Benchmark it via `generic_benchmark.py`.
4. **Record the result in `EXPERIMENTS.md`** — regardless of whether it helped.

If a paper is too heavy to implement simply, note that in `EXPERIMENTS.md` (what it would take, why deferred) rather than silently skipping it.

## `EXPERIMENTS.md` — the experiment report (MANDATORY upkeep)

**After every experiment you run, append an entry to `EXPERIMENTS.md`.** This is the single most important habit in this repo — it is what makes the toolbox reusable across tasks. Do not skip it, even for negative or inconclusive results (those are often the most valuable).

Each entry should capture:
- **Method** — which file / technique, and the paper it came from if any.
- **Setup** — model(s), dataset slice, key hyperparameters (n, temperature, max_steps…), backend.
- **Results** — accuracy, avg reward, and cost proxy (tool calls / steps, tokens, wall-clock, $ if API).
- **Verdict** — did it beat the baseline? By how much, at what added cost?
- **Notes for the next task** — would this technique likely transfer? Caveats, failure modes, ideas to try next.

Keep older entries; never rewrite history. The report is append-only.

**Keep `EXPERIMENTS.md` simple and easily readable by a human.** It is a lab
notebook, not a data dump: short prose, small tables, plain numbers. Anyone
should be able to skim it and understand what was tried and what happened without
running any code. Favor clarity over exhaustiveness — if raw output is bulky, put
it in `data/` and link to it rather than pasting it in.

## Running things

**Benchmark any toolbox method (task-agnostic harness):**
```bash
python generic_benchmark.py   # edit the import to select the method
```

**Local MLX baseline, single episode (Apple Silicon):**
```bash
python run_hscode.py
```

**Sweep the MLX baseline across Qwen3-4B/8B/14B (checkpoints, resumable):**
```bash
python benchmark.py
```

**GRPO fine-tuning (training-based method, needs CUDA + BitsAndBytes):**
```bash
python -m training_methods.grpo_hscode        # locally, from the repo root
# or on an A100 via Google Cloud Batch:
PROJECT_ID=<proj> ./google_gpu/submit_grpo.sh
```

## The HS code task bench: `hscode_env.py`

`HSCodeEnv` is the stateful tool-calling environment for the first task. Tools exposed to the model:
- `search_section_children(section_letter)` — 2-digit chapters in a section (A–U).
- `search_code_children(code)` — children of a 2- or 4-digit code.
- `submit_final_code(code)` — lock in the 6-digit answer.

**Key constraint**: a code is only valid to search or submit if a prior tool call returned it — the env forbids hallucinated codes. The env also tracks a graded `reward` (positive for correct navigation and submission, negative per tool call to penalize wandering) that doubles as the GRPO training signal.

### Data (`data/`)
- `harmonized_system_by_parent.csv` — the HS tree indexed by parent group; queried live during episodes.
- `sections_prepared.csv` — section letters (A–U) → names; injected into the system prompt.
- `benchmark_dataset.csv` — columns: `product_description`, `answer` (6-digit), `hs_2`, `hs_4`, `section`, `trickiness`.
- `benchmark_<model>.csv` — per-model results from `benchmark.py`.

## Adding a new task (the eventual goal)

When HS codes are exhausted and we move to a new task, the pattern to follow:
1. Write a new task bench (the analog of `hscode_env.py`) — its data, tools, and a correctness/reward signal.
2. Repoint the toolbox methods at it (they should need only their task-specific env swapped).
3. Replay the toolbox, guided by `EXPERIMENTS.md`, starting with whatever transferred best on prior tasks.
4. Log everything back into `EXPERIMENTS.md`.
