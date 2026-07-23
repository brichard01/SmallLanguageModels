"""Step 2 of distillation: train the Qwen3-4B student on the teacher dataset.

Consumes the dataset produced by `distill_generate.py` (one row per teacher
assistant turn, carrying `prompt_ids`, `output_ids`, and per-output-token
top-k `topk_ids` / `topk_logprobs`) and trains a QLoRA student with the loss

    L = alpha * CE(student, teacher_sampled_token)
        + (1 - alpha) * T**2 * KL( p_teacher_topk || q_student )

evaluated only on the teacher-generated (output) tokens of each turn.

- CE ("hard"): student predicts the teacher's actually-sampled next token.
- KL ("soft"/dark knowledge): student matches the teacher's top-k distribution,
  restricted to the teacher's top-k support and renormalized, with temperature T.

Teacher (Qwen3-32B) and student (Qwen3-4B) share the exact same tokenizer/vocab,
so teacher token ids index straight into the student's logits — no remapping.

Run (on CUDA; reuses the google_gpu grpo image, which has torch/transformers/
peft/bitsandbytes/datasets), from the repo root:
    python -m training_methods.distill_train \
        --dataset brichard01/hscode-distill-qwen3-32b \
        --output-dir ./qwen3-4b-hscode-distill
"""

import argparse
import os

import torch
import torch.nn.functional as F

PAD_LP = -1e30  # sentinel for "no teacher token here" (Arrow can't store -inf)


def _empty():
    # A complete but empty example so datasets.map keeps a consistent schema for
    # skipped rows (they are dropped afterwards by filtering on input_ids).
    return {"input_ids": [], "loss_mask": [], "ce_target": [], "kd_ids": [], "kd_lp": []}


def build_example(row, topk, max_len):
    """Turn one teacher turn into student training arrays (or an empty example if
    the turn is degenerate / too long).

    input_ids = prompt_ids + output_ids. The student's logits at position i predict
    input_ids[i+1], so the output token with output-index j (absolute position
    L+j) is predicted from logits position L+j-1. We attach the teacher's top-k for
    output token j to that same logits position."""
    prompt_ids = list(row["prompt_ids"])
    output_ids = list(row["output_ids"])
    tk_ids = row["topk_ids"]
    tk_lp = row["topk_logprobs"]

    L, M = len(prompt_ids), len(output_ids)
    if M == 0:
        return _empty()
    input_ids = prompt_ids + output_ids
    N = len(input_ids)
    if N > max_len:
        return _empty()

    loss_mask = [0] * N
    ce_target = [0] * N
    kd_ids = [[0] * topk for _ in range(N)]
    kd_lp = [[PAD_LP] * topk for _ in range(N)]

    for j in range(M):
        i = L + j - 1  # logits position that predicts output token j
        loss_mask[i] = 1
        ce_target[i] = output_ids[j]
        ids_j = list(tk_ids[j])[:topk]
        lp_j = list(tk_lp[j])[:topk]
        for c in range(len(ids_j)):
            kd_ids[i][c] = int(ids_j[c])
            kd_lp[i][c] = float(lp_j[c])

    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "ce_target": ce_target,
        "kd_ids": kd_ids,
        "kd_lp": kd_lp,
    }


class Collator:
    """Right-pad a batch of build_example() dicts into tensors."""

    def __init__(self, pad_id, topk):
        self.pad_id = pad_id
        self.topk = topk

    def __call__(self, feats):
        N = max(len(f["input_ids"]) for f in feats)
        B, K = len(feats), self.topk
        input_ids = torch.full((B, N), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, N), dtype=torch.long)
        loss_mask = torch.zeros((B, N), dtype=torch.bool)
        ce_target = torch.zeros((B, N), dtype=torch.long)
        kd_ids = torch.zeros((B, N, K), dtype=torch.long)
        kd_lp = torch.full((B, N, K), PAD_LP, dtype=torch.float)
        for b, f in enumerate(feats):
            n = len(f["input_ids"])
            input_ids[b, :n] = torch.tensor(f["input_ids"], dtype=torch.long)
            attn[b, :n] = 1
            loss_mask[b, :n] = torch.tensor(f["loss_mask"], dtype=torch.bool)
            ce_target[b, :n] = torch.tensor(f["ce_target"], dtype=torch.long)
            kd_ids[b, :n] = torch.tensor(f["kd_ids"], dtype=torch.long)
            kd_lp[b, :n] = torch.tensor(f["kd_lp"], dtype=torch.float)
        return {
            "input_ids": input_ids, "attention_mask": attn, "loss_mask": loss_mask,
            "ce_target": ce_target, "kd_ids": kd_ids, "kd_lp": kd_lp,
        }


def kd_loss(logits, batch, alpha, T):
    """alpha*CE + (1-alpha)*T^2*KL over teacher top-k, on output positions only."""
    pos = batch["loss_mask"]                       # [B,N]
    sel = logits[pos].float()                       # [P,V]
    if sel.numel() == 0:
        return logits.sum() * 0.0                    # empty batch guard (keeps grad)
    tgt = batch["ce_target"][pos]                    # [P]
    tk_ids = batch["kd_ids"][pos]                    # [P,K]
    tk_lp = batch["kd_lp"][pos]                       # [P,K]

    ce = F.cross_entropy(sel, tgt)

    valid = tk_lp > (PAD_LP / 2)                     # True where a teacher token exists
    # teacher distribution over its top-k support (renormalized), temperature T
    t_logits = (tk_lp / T).masked_fill(~valid, float("-inf"))
    p_teacher = torch.softmax(t_logits, dim=-1)      # [P,K]
    # student log-probs over the full vocab, gathered at the teacher's top-k ids,
    # then renormalized over that same support
    student_logp = F.log_softmax(sel / T, dim=-1)    # [P,V]
    s_tk = torch.gather(student_logp, 1, tk_ids.clamp(min=0))  # [P,K]
    q_logp = torch.log_softmax(s_tk.masked_fill(~valid, float("-inf")), dim=-1)
    logp_teacher = torch.log(p_teacher.clamp_min(1e-12))
    kl = (p_teacher * (logp_teacher - q_logp)).masked_fill(~valid, 0.0).sum(-1)  # [P]
    kd = (T * T) * kl.mean()

    return alpha * ce + (1.0 - alpha) * kd


def main():
    ap = argparse.ArgumentParser(description="Distill Qwen3-4B from teacher top-k logprobs.")
    ap.add_argument("--dataset", default="brichard01/hscode-distill-qwen3-32b",
                    help="HF hub repo id, or a local save_from_disk dir")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--output-dir", default="./qwen3-4b-hscode-distill")
    ap.add_argument("--push-repo", default=None, help="if set, push the adapter here")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on the hard CE term")
    ap.add_argument("--kd-temperature", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=4096, help="skip turns longer than this")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="only first N rows (smoke test)")
    args = ap.parse_args()

    from datasets import load_dataset, load_from_disk
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # --- data ---
    if os.path.isdir(args.dataset):
        ds = load_from_disk(args.dataset)
    else:
        ds = load_dataset(args.dataset, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    raw_cols = ds.column_names
    n_before = len(ds)
    ds = ds.map(lambda r: build_example(r, args.topk, args.max_len),
                remove_columns=raw_cols)
    ds = ds.filter(lambda r: len(r["input_ids"]) > 0)
    print(f"Training turns: {len(ds)} (skipped {n_before - len(ds)} empty/too-long)")

    # --- model (QLoRA) ---
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    class DistillTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            loss = kd_loss(out.logits, inputs, args.alpha, args.kd_temperature)
            return (loss, out) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_steps=200,
        report_to="none",
        remove_unused_columns=False,  # our collator needs the custom columns
    )
    trainer = DistillTrainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=Collator(tokenizer.pad_token_id, args.topk),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved adapter to {args.output_dir}")

    if args.push_repo:
        repo = args.push_repo
        if "/" not in repo:
            from huggingface_hub import whoami
            repo = f"{whoami(token=os.environ.get('HF_TOKEN'))['name']}/{repo}"
        model.push_to_hub(repo)
        tokenizer.push_to_hub(repo)
        print(f"Pushed adapter to hub: {repo}")


if __name__ == "__main__":
    main()
