# This utility script was generated with assistance from ChatGPT.

import csv
import json
import re
from pathlib import Path
from statistics import mean, median


RESULT_ROOT = Path("results/task1/linear_q_agent")
OUTPUT_CSV = RESULT_ROOT / "task1_summary.csv"

RUN_PATTERN = re.compile(
    r"^(f[01])_(sparse|basic|shaped)_seed(\d+)$"
)

TOTAL_COINS = 50
MAX_STEPS = 400
FINAL_TRAIN_WINDOW = 100


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def round_number(name):
    """
    Example:
    'Round 01 (2026-09-11 ...)' -> 1
    'Round 600 (...)' -> 600
    """
    match = re.search(r"Round\s+(\d+)", name)
    if match is None:
        raise ValueError(f"Could not parse round number from: {name}")
    return int(match.group(1))


def extract_rounds(stats):
    rounds = []

    for name, data in stats["by_round"].items():
        rounds.append(
            {
                "round": round_number(name),
                "coins": data["coins"],
                "steps": data["steps"],
            }
        )

    return sorted(rounds, key=lambda x: x["round"])


def analyze_run(run_dir):
    match = RUN_PATTERN.match(run_dir.name)

    if match is None:
        return None

    feature_mode, reward_mode, seed = match.groups()

    train_path = run_dir / "train.json"
    eval_path = run_dir / "eval.json"

    if not train_path.exists() or not eval_path.exists():
        print(f"Skipping incomplete run: {run_dir.name}")
        return None

    train_stats = load_json(train_path)
    eval_stats = load_json(eval_path)

    train_rounds = extract_rounds(train_stats)
    eval_rounds = extract_rounds(eval_stats)

    train_coins = [r["coins"] for r in train_rounds]
    final_train_coins = [
        r["coins"] for r in train_rounds[-FINAL_TRAIN_WINDOW:]
    ]

    eval_coins = [r["coins"] for r in eval_rounds]

    completed = [
        r for r in eval_rounds
        if r["coins"] >= TOTAL_COINS
    ]

    timed_out = [
        r for r in eval_rounds
        if r["steps"] >= MAX_STEPS
        and r["coins"] < TOTAL_COINS
    ]

    completed_steps = [
        r["steps"] for r in completed
    ]

    return {
        "feature": feature_mode.upper(),
        "reward": reward_mode,
        "seed": int(seed),

        "train_avg_coins": mean(train_coins),
        "train_final100_avg_coins": mean(final_train_coins),

        "eval_avg_coins": mean(eval_coins),
        "eval_median_coins": median(eval_coins),

        "completion_rate": len(completed) / len(eval_rounds),
        "timeout_rate": len(timed_out) / len(eval_rounds),

        "completed_avg_steps":
            mean(completed_steps)
            if completed_steps else None,
    }


def print_table(results):
    headers = [
        "Feature",
        "Reward",
        "Seed",
        "Train avg",
        "Final 100",
        "Eval avg",
        "Median",
        "Completion",
        "Timeout",
        "Avg steps*",
    ]

    rows = []

    for r in results:
        rows.append(
            [
                r["feature"],
                r["reward"],
                str(r["seed"]),
                f'{r["train_avg_coins"]:.2f}',
                f'{r["train_final100_avg_coins"]:.2f}',
                f'{r["eval_avg_coins"]:.2f}',
                f'{r["eval_median_coins"]:.1f}',
                f'{100 * r["completion_rate"]:.1f}%',
                f'{100 * r["timeout_rate"]:.1f}%',
                (
                    f'{r["completed_avg_steps"]:.1f}'
                    if r["completed_avg_steps"] is not None
                    else "-"
                ),
            ]
        )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def format_row(row):
        return " | ".join(
            value.ljust(widths[i])
            for i, value in enumerate(row)
        )

    print()
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))

    for row in rows:
        print(format_row(row))

    print()
    print("* Avg steps among completed evaluation rounds only")


def save_csv(results):
    fieldnames = [
        "feature",
        "reward",
        "seed",
        "train_avg_coins",
        "train_final100_avg_coins",
        "eval_avg_coins",
        "eval_median_coins",
        "completion_rate",
        "timeout_rate",
        "completed_avg_steps",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved CSV: {OUTPUT_CSV}")


def main():
    if not RESULT_ROOT.exists():
        raise FileNotFoundError(
            f"Result directory not found: {RESULT_ROOT}"
        )

    results = []

    for run_dir in RESULT_ROOT.iterdir():
        if not run_dir.is_dir():
            continue

        result = analyze_run(run_dir)

        if result is not None:
            results.append(result)

    reward_order = {
        "sparse": 0,
        "basic": 1,
        "shaped": 2,
    }

    results.sort(
        key=lambda r: (
            r["feature"],
            reward_order[r["reward"]],
            r["seed"],
        )
    )

    if not results:
        print("No complete Task 1 experiment results found.")
        return

    print_table(results)
    save_csv(results)


if __name__ == "__main__":
    main()