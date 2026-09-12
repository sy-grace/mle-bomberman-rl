# This utility script was generated with assistance from ChatGPT.

import csv
import json
import re
from pathlib import Path
from statistics import mean, median, stdev


SEEDS = (123, 456, 2026)

# Existing formal SARSA(lambda=0.8) Task 1 results.
LAMBDA_08_ROOT = Path("results/task1/sarsa_lambda_agent")

# New SARSA(0), lambda=0.0, ablation results.
LAMBDA_00_ROOT = Path("results/task1_lambda_ablation/sarsa_lambda_agent")

RUNS_CSV = Path("docs/experiments/task1_sarsa_lambda_ablation_runs.csv")
SUMMARY_CSV = Path("docs/experiments/task1_sarsa_lambda_ablation_summary.csv")

TOTAL_COINS = 50
MAX_STEPS = 400
FINAL_TRAIN_WINDOW = 100

METRICS = [
    "train_avg_coins",
    "train_final100_avg_coins",
    "eval_avg_coins",
    "eval_median_coins",
    "completion_rate",
    "timeout_rate",
    "completed_avg_steps",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def round_number(name):
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


def analyze_run(lambda_value, seed, run_dir):
    train_path = run_dir / "train.json"
    eval_path = run_dir / "eval.json"

    if not train_path.exists() or not eval_path.exists():
        print(f"Skipping incomplete run: {run_dir}")
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

    completed_steps = [r["steps"] for r in completed]

    return {
        "lambda": lambda_value,
        "seed": seed,
        "train_avg_coins": mean(train_coins),
        "train_final100_avg_coins": mean(final_train_coins),
        "eval_avg_coins": mean(eval_coins),
        "eval_median_coins": median(eval_coins),
        "completion_rate": len(completed) / len(eval_rounds),
        "timeout_rate": len(timed_out) / len(eval_rounds),
        "completed_avg_steps": (
            mean(completed_steps)
            if completed_steps else None
        ),
    }


def collect_results():
    results = []

    for seed in SEEDS:
        lambda_08_dir = LAMBDA_08_ROOT / f"f1_basic_seed{seed}"
        result = analyze_run(0.8, seed, lambda_08_dir)
        if result is not None:
            results.append(result)

        lambda_00_dir = LAMBDA_00_ROOT / f"lambda0.0_seed{seed}"
        result = analyze_run(0.0, seed, lambda_00_dir)
        if result is not None:
            results.append(result)

    return sorted(results, key=lambda r: (r["lambda"], r["seed"]))


def sample_std(values):
    return stdev(values) if len(values) >= 2 else None


def aggregate_results(results):
    summaries = []

    for lambda_value in sorted({r["lambda"] for r in results}):
        runs = [r for r in results if r["lambda"] == lambda_value]

        row = {
            "lambda": lambda_value,
            "n_seeds": len(runs),
            "seeds": ",".join(str(r["seed"]) for r in runs),
        }

        for metric in METRICS:
            values = [
                r[metric]
                for r in runs
                if r[metric] is not None
            ]

            row[f"{metric}_mean"] = mean(values) if values else None
            row[f"{metric}_std"] = sample_std(values) if values else None

        summaries.append(row)

    return summaries


def format_mean_std(mean_value, std_value, decimals=2, percent=False):
    if mean_value is None:
        return "-"

    if percent:
        mean_value *= 100
        if std_value is not None:
            std_value *= 100

    suffix = "%" if percent else ""

    if std_value is None:
        return f"{mean_value:.{decimals}f}{suffix}"

    return (
        f"{mean_value:.{decimals}f} +/- "
        f"{std_value:.{decimals}f}{suffix}"
    )


def print_text_table(title, headers, rows):
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
    print(title)
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))

    for row in rows:
        print(format_row(row))


def print_run_table(results):
    headers = [
        "Lambda",
        "Seed",
        "Train avg",
        "Final 100",
        "Eval avg",
        "Median",
        "Completion",
        "Timeout",
        "Completed steps",
    ]

    rows = []

    for r in results:
        completed_steps = (
            f'{r["completed_avg_steps"]:.1f}'
            if r["completed_avg_steps"] is not None
            else "-"
        )

        rows.append(
            [
                f'{r["lambda"]:.1f}',
                str(r["seed"]),
                f'{r["train_avg_coins"]:.2f}',
                f'{r["train_final100_avg_coins"]:.2f}',
                f'{r["eval_avg_coins"]:.2f}',
                f'{r["eval_median_coins"]:.1f}',
                f'{100 * r["completion_rate"]:.1f}%',
                f'{100 * r["timeout_rate"]:.1f}%',
                completed_steps,
            ]
        )

    print_text_table("Per-seed lambda ablation results", headers, rows)


def print_summary_table(summaries):
    headers = [
        "Lambda",
        "n",
        "Train avg",
        "Final 100",
        "Eval avg",
        "Median",
        "Completion",
        "Timeout",
        "Completed steps",
    ]

    rows = []

    for r in summaries:
        rows.append(
            [
                f'{r["lambda"]:.1f}',
                str(r["n_seeds"]),
                format_mean_std(
                    r["train_avg_coins_mean"],
                    r["train_avg_coins_std"],
                ),
                format_mean_std(
                    r["train_final100_avg_coins_mean"],
                    r["train_final100_avg_coins_std"],
                ),
                format_mean_std(
                    r["eval_avg_coins_mean"],
                    r["eval_avg_coins_std"],
                ),
                format_mean_std(
                    r["eval_median_coins_mean"],
                    r["eval_median_coins_std"],
                    decimals=1,
                ),
                format_mean_std(
                    r["completion_rate_mean"],
                    r["completion_rate_std"],
                    decimals=1,
                    percent=True,
                ),
                format_mean_std(
                    r["timeout_rate_mean"],
                    r["timeout_rate_std"],
                    decimals=1,
                    percent=True,
                ),
                format_mean_std(
                    r["completed_avg_steps_mean"],
                    r["completed_avg_steps_std"],
                    decimals=1,
                ),
            ]
        )

    print_text_table(
        "Aggregated lambda ablation results (mean +/- sample std across seeds)",
        headers,
        rows,
    )


def save_runs_csv(results):
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "lambda",
        "seed",
        "train_avg_coins",
        "train_final100_avg_coins",
        "eval_avg_coins",
        "eval_median_coins",
        "completion_rate",
        "timeout_rate",
        "completed_avg_steps",
    ]

    with open(RUNS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def save_summary_csv(summaries):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "lambda",
        "n_seeds",
        "seeds",
        "train_avg_coins_mean",
        "train_avg_coins_std",
        "train_final100_avg_coins_mean",
        "train_final100_avg_coins_std",
        "eval_avg_coins_mean",
        "eval_avg_coins_std",
        "eval_median_coins_mean",
        "eval_median_coins_std",
        "completion_rate_mean",
        "completion_rate_std",
        "timeout_rate_mean",
        "timeout_rate_std",
        "completed_avg_steps_mean",
        "completed_avg_steps_std",
    ]

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def main():
    results = collect_results()

    if not results:
        print("No complete lambda ablation results found.")
        return

    expected_runs = 2 * len(SEEDS)
    if len(results) != expected_runs:
        print(
            f"Warning: expected {expected_runs} complete runs "
            f"but found {len(results)}."
        )

    summaries = aggregate_results(results)

    print_run_table(results)
    print_summary_table(summaries)

    save_runs_csv(results)
    save_summary_csv(summaries)

    print()
    print(f"Saved run-level results: {RUNS_CSV}")
    print(f"Saved aggregated summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
