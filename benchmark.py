import json
import os
import mlx_lm
import pandas as pd
import run_hscode
from run_hscode import run_episode

MODELS = [
    "Qwen/Qwen3-4B-MLX-4bit",
    "Qwen/Qwen3-8B-MLX-4bit",
    "Qwen/Qwen3-14B-MLX-4bit",
]

FAILED_RESULT = {"submitted": None, "reward": None, "correct": None, "steps": 0, "messages": None}

df = pd.read_csv("data/benchmark_dataset.csv", dtype=str).reset_index(drop=True)#.sample(1).reset_index(drop=True)

for model_name in MODELS:
    print(f"\n{'#'*60}")
    print(f"Model: {model_name}")
    print('#'*60)

    model_tag = model_name.split("/")[-1]
    output_path = f"data/benchmark_{model_tag}.csv"

    # Load completed rows from existing output file
    results = {}
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path, dtype=str)
        existing_df["steps"] = pd.to_numeric(existing_df["steps"], errors="coerce").fillna(0).astype(int)
        for idx in range(min(len(existing_df), len(df))):
            if existing_df.iloc[idx]["steps"] > 0:
                row = existing_df.iloc[idx]
                results[idx] = {
                    "submitted": row["submitted"],
                    "reward": row["reward"],
                    "correct": row["correct"],
                    "steps": row["steps"],
                    "messages": row["messages"],
                }
        print(f"Loaded {len(results)} completed rows, {len(df) - len(results)} remaining.")

    todo = [i for i in range(len(df)) if i not in results]

    if not todo:
        print("All rows already complete, skipping model load.")
    else:
        run_hscode.model, run_hscode.tokenizer = mlx_lm.load(model_name)

        for i in todo:
            row = df.iloc[i]
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(df)}] {row['product_description'][:80]}")
            print('='*60)
            try:
                result = run_episode(
                    product_description=row["product_description"],
                    answer=row["answer"],
                    hs_2=row["hs_2"],
                    hs_4=row["hs_4"],
                    section=row["section"],
                )
                result["messages"] = json.dumps(result["messages"])
            except Exception as e:
                print(f"Episode failed: {e}")
                result = dict(FAILED_RESULT)
            results[i] = result

            # Checkpoint: save after every completed row
            result_rows = [results.get(j, dict(FAILED_RESULT)) for j in range(len(df))]
            output_df = pd.concat([df, pd.DataFrame(result_rows)], axis=1)
            output_df.to_csv(output_path, index=False)

    # Stats from saved file
    final_df = pd.read_csv(output_path, dtype=str)
    completed = pd.to_numeric(final_df["steps"], errors="coerce").fillna(0).gt(0).sum()
    correct = (final_df["correct"] == "True").sum()
    print(f"\nSaved to {output_path}")
    print(f"Accuracy: {correct/completed:.1%} ({correct}/{completed})" if completed else "No completed episodes.")
