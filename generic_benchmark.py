import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def benchmark(fn, path="data/benchmark_dataset.csv", workers=8):
    rows = pd.read_csv(path, dtype=str).head(90).to_dict("records")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fn, rows))

    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_reward = sum(r["reward"] for r in results) / len(results)
    print(f"n={len(results)}  accuracy={accuracy:.1%}  avg_reward={avg_reward:.2f}")
    return results


if __name__ == "__main__":
    from inference_methods.raw_openai import run
    #from inference_methods.self_consistency import run
    #from inference_methods.inspect_submit import run
    #from inference_methods.self_refine import run
    benchmark(run)
