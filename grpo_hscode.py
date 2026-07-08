import pandas as pd
import random
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from hscode_env import HSCodeEnv, SYSTEM_PROMPT

random.seed(42)


def reward_env(environments, **kwargs):
    rewards = []
    for env in environments:
        reward = env.reward
        rewards.append(reward)
    print(rewards)
    return rewards


def build_user_prompt(product_description: str) -> str:
    return f"""{product_description}

Use the tools to navigate the code hierarchy and then submit your final answer"""


def build_grpo_row(row):
    return {
        "prompt": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_prompt(str(row["product_description"])),
            },
        ],
        "answer": row['hs_6'],
        "hs_4": row['hs_4'],
        "hs_2": row['hs_2'],
        "section": row['section_letter']
    }


if __name__ == "__main__":
    hscode_dataset = pd.read_csv("data/HSCode_full.csv")
    hscode_dataset.hs_6 = hscode_dataset.hs_6.astype(str)

    rows = hscode_dataset.to_dict("records")
    random.shuffle(rows)

    split_idx = int(len(rows) * 0.75)
    train_rows_raw = rows[:split_idx]
    eval_rows_raw = rows[split_idx:]

    train_rows = [build_grpo_row(row) for row in train_rows_raw]
    eval_rows = [build_grpo_row(row) for row in eval_rows_raw]

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)

    model_name = "Qwen/Qwen3-4B"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",  # remove if flash-attn is not installed
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    grpo_config = GRPOConfig(
        output_dir="./qwen3-4b-hscode-grpo-qlora",

        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,

        max_completion_length=2048,

        learning_rate=5e-6,
        beta=0.02,

        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        logging_steps=10,
        save_steps=100,
        num_train_epochs=1,

        use_vllm=False,
        report_to="none",

        chat_template_kwargs={"enable_thinking": True}
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        environment_factory=HSCodeEnv,
        reward_funcs=[
            reward_env
        ],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
