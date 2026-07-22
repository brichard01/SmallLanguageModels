"""Self-consistency method: run the baseline n times and majority-vote the code."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from methods.raw_openai import run as run_once


def run(row, n=5, model="gpt-5-nano", max_steps=15):
    with ThreadPoolExecutor(max_workers=n) as pool:
        runs = list(pool.map(lambda _: run_once(row, model=model, temperature=1, max_steps=max_steps), range(n)))

    votes = Counter(r["submitted"] for r in runs if r["submitted"])
    winner = votes.most_common(1)[0][0] if votes else None

    # Report the vote-winning code, backed by one run that produced it.
    chosen = next((r for r in runs if r["submitted"] == winner), runs[0])
    return {
        "submitted": winner,
        "reward": chosen["reward"],
        "correct": winner == row["answer"],
        "steps": sum(r["steps"] for r in runs),
        "votes": dict(votes),
    }


if __name__ == "__main__":
    import pandas as pd

    row = pd.read_csv("data/benchmark_dataset.csv", dtype=str).to_dict("records")[0]
    print("Product:", row["product_description"])
    print("Answer:", row["answer"])
    print("Result:", run(row))
