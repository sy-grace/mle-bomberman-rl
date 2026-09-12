# This utility script was generated with assistance from ChatGPT.

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev


SUPPORTED_AGENTS = {
    "linear_q_agent": "linear_q",
    "sarsa_lambda_agent": "sarsa_lambda",
}


def experiment_paths(agent):
    """Return the result directory and output CSV paths for one agent."""
    output_name = SUPPORTED_AGENTS[agent]

    result_root = Path("results/task1") / agent
    runs_csv = Path(f"docs/experiments/task1_{output_name}_runs.csv")
    summary_csv = Path(f"docs/experiments/task1_{output_name}_summary.csv")

    return result_root, runs_csv, summary_csv

RUN_PATTERN = re.compile(
    r"^(f[01])_(sparse|basic|shaped)_seed(\d+)$"
)

TOTAL_COINS = 50
MAX_STEPS = 400
FINAL_TRAIN_WINDOW = 100

REWARD_ORDER = {
    "sparse": 0,
    "basic": 1,
    "shaped": 2,
}

METRICS = [
    "train_avg_coins",
    "train_final100_avg_coins",
    "eval_avg_coins",
    "eval_median_coins",
    "completion_rate",
    "timeout_rate",
]


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

    completed_steps = [r["steps"] for r in completed]

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
        "completed_avg_steps": (
            mean(completed_steps)
            if completed_steps else None
        ),
    }


def sample_std(values):
    """Return sample standard deviation, or None if fewer than 2 values."""
    return stdev(values) if len(values) >= 2 else None


def aggregate_results(results):
    grouped = defaultdict(list)

    for result in results:
        key = (result["feature"], result["reward"])
        grouped[key].append(result)

    summaries = []

    for (feature, reward), runs in grouped.items():
        row = {
            "feature": feature,
            "reward": reward,
            "n_seeds": len(runs),
            "seeds": ",".join(
                str(r["seed"])
                for r in sorted(runs, key=lambda r: r["seed"])
            ),
        }

        for metric in METRICS:
            values = [r[metric] for r in runs]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = sample_std(values)

        completed_steps = [
            r["completed_avg_steps"]
            for r in runs
            if r["completed_avg_steps"] is not None
        ]

        row["completed_avg_steps_mean"] = (
            mean(completed_steps)
            if completed_steps else None
        )
        row["completed_avg_steps_std"] = (
            sample_std(completed_steps)
            if completed_steps else None
        )

        summaries.append(row)

    summaries.sort(
        key=lambda r: (
            r["feature"],
            REWARD_ORDER[r["reward"]],
        )
    )

    return summaries


def format_mean_std(mean_value, std_value, decimals=2, percent=False):
    if percent:
        mean_value *= 100
        if std_value is not None:
            std_value *= 100

    if std_value is None:
        suffix = "%" if percent else ""
        return f"{mean_value:.{decimals}f}{suffix}"

    suffix = "%" if percent else ""
    return (
        f"{mean_value:.{decimals}f} +/- "
        f"{std_value:.{decimals}f}{suffix}"
    )


def print_run_table(results):
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
            ]
        )

    print_text_table("Per-seed results", headers, rows)


def print_summary_table(summaries):
    headers = [
        "Feature",
        "Reward",
        "n",
        "Train avg",
        "Final 100",
        "Eval avg",
        "Median",
        "Completion",
        "Timeout",
    ]

    rows = []

    for r in summaries:
        rows.append(
            [
                r["feature"],
                r["reward"],
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
            ]
        )

    print_text_table(
        "Aggregated results (mean +/- sample std across seeds)",
        headers,
        rows,
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


def save_runs_csv(results, runs_csv):
    runs_csv.parent.mkdir(parents=True, exist_ok=True)

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

    with open(runs_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def save_summary_csv(summaries, summary_csv):
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "feature",
        "reward",
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

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Task 1 experiment results for one agent."
    )
    parser.add_argument(
        "--agent",
        choices=sorted(SUPPORTED_AGENTS),
        default="linear_q_agent",
        help="Agent result directory to summarize (default: linear_q_agent).",
    )
    args = parser.parse_args()

    result_root, runs_csv, summary_csv = experiment_paths(args.agent)

    if not result_root.exists():
        raise FileNotFoundError(
            f"Result directory not found: {result_root}"
        )

    results = []

    for run_dir in result_root.iterdir():
        if not run_dir.is_dir():
            continue

        result = analyze_run(run_dir)

        if result is not None:
            results.append(result)

    results.sort(
        key=lambda r: (
            r["feature"],
            REWARD_ORDER[r["reward"]],
            r["seed"],
        )
    )

    if not results:
        print(f"No complete Task 1 experiment results found for {args.agent}.")
        return

    summaries = aggregate_results(results)

    print(f"Agent: {args.agent}")
    print_run_table(results)
    print_summary_table(summaries)

    save_runs_csv(results, runs_csv)
    save_summary_csv(summaries, summary_csv)

    print()
    print(f"Saved run-level results: {runs_csv}")
    print(f"Saved aggregated summary: {summary_csv}")


if __name__ == "__main__":
    main()
