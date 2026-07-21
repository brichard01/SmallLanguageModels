# EXPERIMENTS.md — Experiment report

Running, **append-only** log of every method tried and what it produced. This is
the memory of the toolbox: it lets us replay the winning techniques on the next
task and skip the dead ends. See `CLAUDE.md` for why this file matters.

**Keep this file simple and readable by a human.** Short prose, small tables,
plain numbers. Anyone should be able to skim it and get the story without running
code. **Add an entry after every experiment — including negative or inconclusive
ones.** A method that loses on HS codes still belongs in the toolbox; the entry
records why it lost and whether it might transfer to another task.

## How to log an entry

Copy the template, fill it in, append under "Experiment log" (newest at the
bottom). Do not edit or delete past entries.

```markdown
### <YYYY-MM-DD> — <method name>

- **Method / technique:** <file in methods/, and the paper if any>
- **Setup:** model(s), dataset slice, hyperparams (n, temperature, max_steps…), backend
- **Results:** accuracy = __%, avg reward = __, cost = __ (steps/tokens/wall-clock/$)
- **Baseline compared to:** <which run, its accuracy>
- **Verdict:** beat baseline? by how much, at what added cost?
- **Transferability / notes:** would this help on another task? failure modes, next ideas
```

## Current task bench

**HS code classification** — a strictly hierarchical standard for classifying
goods (customs/tax). Structure: Section > Chapter (2 digits) > Heading (4 digits)
> Subheading (6 digits). The model must navigate to the exact 6-digit code; any
error has real financial and logistical consequences. Bench: `hscode_env.py`,
dataset `data/benchmark_dataset.csv`. Metrics: accuracy (exact 6-digit match) and
avg env reward.

## Reference points (non-agentic, large models)

Baselines with no tools, to know what "raw model knowledge" is worth:

| Setup | Data | Accuracy |
|---|---|---|
| GPT-4.1, no tools | public `ATH-MaaS/HSCodeComp` | 42.9% |
| GPT-5, no tools | cleaned dataset (see note) | 68.7% |

_Note:_ the public dataset had quality issues and inconsistencies, which capped
the GPT-4.1 number. A cleaned dataset was generated to get a fairer read (GPT-5,
68.7%). The cleaned set is the basis for the agentic work below.

## Headline result

The agentic approach (SLM + hierarchy-search tools + guardrails that forbid
proposing any code not returned by a tool) lets small local models **beat the
non-agentic large-model reference**: Qwen3-14B reaches 90% on a 20-case sample —
above GPT-5's 68.7% without tools. Tools + structure > raw model size, for this
task.

## Methods available (toolbox inventory)

| Method | File | Status | Best result so far |
|---|---|---|---|
| Agentic baseline (raw tool-calling) | `methods/raw_openai.py` | implemented | 72.2% (GPT-5-nano, n=90) |
| Self-consistency (majority vote) | `methods/self_consistency.py` | implemented | 75.6% (GPT-5-nano, 5 votes, n=90) |
| Submit-then-verify (self-critique) | `methods/inspect_submit.py` | implemented | no gain yet |
| Self-Refine (feedback→refine loop) | `methods/self_refine.py` | implemented | 81.1% (GPT-5-nano, n=90) |
| DSPy GEPA (prompt optimization) | _not in repo_ | tried | marginal, dropped |
| Local MLX agentic (Qwen3 4B/8B/14B) | `run_hscode.py` + `benchmark.py` | implemented | 90% (14B, n=20) |
| GRPO / QLoRA fine-tuning | `grpo_hscode.py` | implemented | not yet run |

Keep this table in sync as methods are added and benchmarked.

---

## Experiment log

### Agentic SLM baseline — local MLX (Qwen3, n=20)

- **Method / technique:** agentic navigation of the HS hierarchy with search
  tools + guardrail forbidding any code not returned by a tool (`run_hscode.py`,
  `hscode_env.py`).
- **Setup:** Qwen3 4B / 8B / 14B (MLX 4-bit, Apple Silicon), 20-case sample.
- **Results:**
  | Model | Accuracy |
  |---|---|
  | Qwen3-14B-MLX-4bit | 90.0% |
  | Qwen3-8B-MLX-4bit | 85.0% |
  | Qwen3-4B-MLX-4bit | 75.0% |
- **Baseline compared to:** GPT-5 non-agentic (68.7%). All three SLMs with tools
  beat it except the 4B, which is close.
- **Verdict:** strong. The agentic scaffold is the single biggest lever so far —
  small local models beat a large model that lacks tools.
- **Transferability / notes:** the guardrail (only act on tool-returned values)
  kills hallucination and should transfer to any tool-navigation task. Clear size
  trend: bigger = better. **Key failure mode of small models: lack of hindsight
  — the smaller the model, the less it questions itself. An early reasoning error
  tends to stick; it stays on its initial track without self-correcting. The 14B
  is noticeably better at catching its own mistakes mid-episode.** This motivates
  the self-correction methods below.

### DSPy GEPA — prompt optimization

- **Method / technique:** DSPy GEPA automatic prompt optimization for general
  guidelines. (Not committed to the repo.)
- **Setup:** applied to the agentic pipeline.
- **Results:** marginal gain only.
- **Verdict:** not worth it here. A well-written hand-crafted prompt performs as
  well. The HS search space is too vast for optimized general guidelines to
  memorize per-category specifics.
- **Transferability / notes:** likely more useful on tasks with a *smaller,
  learnable* set of rules than HS codes. Keep in the toolbox for such tasks;
  don't reach for it when the label space is huge and open-ended.

### GPT-5-nano agentic baseline (n=90)

- **Method / technique:** agentic baseline, raw tool-calling loop
  (`methods/raw_openai.py`).
- **Setup:** GPT-5-nano, 90 examples, single completion.
- **Results:** accuracy = 72.2%.
- **Verdict:** the reference point that self-consistency and self-critique below
  are measured against.
- **Transferability / notes:** the reusable `run(row)` baseline every other
  method builds on.

### Self-consistency — majority vote (n=90)

- **Method / technique:** run the baseline n times at temperature > 0, majority-
  vote the submitted code (`methods/self_consistency.py`).
- **Setup:** GPT-5-nano, 5 completions, temp > 0, 90 examples.
- **Results:** accuracy = 75.6% (vs 72.2% baseline).
- **Verdict:** modest but real gain (+3.4 pts) at ~5× inference cost. Works by
  reducing variance from stochastic errors.
- **Transferability / notes:** general, task-agnostic, always worth trying early;
  the trade-off is linear cost in n. Good default for any task where a single
  answer is submitted.

### Iterative self-critique — submit-then-verify

- **Method / technique:** after a submission, systematically ask the model to
  explore alternative branches before finalizing (`methods/inspect_submit.py`).
- **Setup:** GPT-5-nano, agentic.
- **Results:** no improvement to note so far.
- **Verdict:** inconclusive / no gain yet. The model tends to systematically
  validate its own initial reasoning rather than genuinely reconsider — the same
  "lack of hindsight" seen in the small local models.
- **Transferability / notes:** **kept in the toolbox, not discarded.** The idea
  is sound; the weakness is that self-critique from the *same* model is biased
  toward confirming itself. Next ideas: force exploration of a fixed number of
  alternative branches before allowing `finish`; use a *separate* critic model or
  a higher-temperature critic; make the critique compare concrete sibling codes
  rather than re-reason freely.

### 2026-07-21 — Self-Refine (feedback → refine loop)

- **Method / technique:** Self-Refine (Madaan et al., 2023, arXiv:2303.17651),
  `methods/self_refine.py`. One agentic episode generates a code; the *same* model
  critiques it (naming concrete sibling/branch alternatives + a STOP flag); a fresh
  episode re-navigates from scratch guided by the full history of prior attempts +
  feedback. Loop until STOP: yes or max_iters.
- **Setup:** GPT-5-nano, 90 examples, `max_iters=3`, `max_steps=15`, temperature=1
  (see caveat), backend = OpenAI chat-completions + `HSCodeEnv` tools.
- **Results:** accuracy = **81.1%**, avg reward = 12.67. Cost proxy: up to 1
  generate + 3×(feedback + refine episode) LLM roundtrips per row — the most
  expensive OpenAI method tried so far (each refine is a full fresh episode, not a
  continuation), though the STOP flag ends many rows after iteration 1.
- **Baseline compared to:** GPT-5-nano agentic baseline 72.2%; self-consistency
  (5 votes) 75.6%.
- **Verdict:** **best OpenAI-method result to date — +8.9 pts over baseline, +5.5
  over self-consistency.** Structured feedback that names concrete alternatives and
  then *re-navigates from the start* beats both raw retries and the earlier
  submit-then-verify self-critique — which is telling: the earlier method (which
  showed no gain) let the model validate its existing trace, whereas Self-Refine
  forces a clean re-navigation acting on specific critique. Forcing a fresh start
  seems to be what breaks the "no hindsight" failure mode of small/cheap models.
- **Caveat (comparability):** gpt-5-nano now rejects `temperature != 1`, so this
  run used temperature=1 for both the episodes and the feedback. The baseline /
  self-consistency numbers in this log predate that constraint (baseline default
  was 0.3). Treat the +8.9 pt delta as indicative, not perfectly matched; a
  temperature=1 re-run of the baseline would tighten the comparison.
- **Transferability / notes:** highly general — three prompts, no training, works
  on any task with a self-checkable output. Key design lesson for the next task:
  **make refinement a fresh attempt guided by specific, alternative-naming feedback
  rather than an in-place patch of the prior trace.** Next ideas: use a separate /
  stronger critic model for the feedback step; measure how often STOP fires early
  vs. burning all 3 iters; ablate the "re-navigate from scratch" choice against
  continuing the same trace to confirm it's the active ingredient.
