# This utility script was generated with assistance from ChatGPT.
#
# Summarize Bomberman experiment results for Task 1 and Task 2.
#
# Examples:
#   python scripts/summarize_experiments.py --task 1 --agent linear_q_agent
#   python scripts/summarize_experiments.py --task 1 --agent sarsa_lambda_agent
#   python scripts/summarize_experiments.py --task 2 --agent linear_q_agent

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

REWARD_ORDER = {
    "sparse": 0,
    "basic": 1,
    "shaped": 2,
}

MAX_STEPS = 400
FINAL_TRAIN_WINDOW = 100

# Task 1: coin-heaven contains 50 coins.
TASK1_TOTAL_COINS = 50

TASK1_RUN_PATTERN = re.compile(
    r"^(f[01])_(sparse|basic|shaped)_seed(\d+)$"
)

# Task 2 currently uses F2. Supporting sparse here as well keeps the
# summarizer useful if a sparse control experiment is added later.
TASK2_RUN_PATTERN = re.compile(
    r"^(f2)_(sparse|basic|shaped)_seed(\d+)$"
)


TASK1_METRICS = [
    "train_avg_coins",
    "train_final100_avg_coins",
    "eval_avg_coins",
    "eval_median_coins",
    "completion_rate",
    "timeout_rate",
]

TASK2_METRICS = [
    "train_avg_coins",
    "train_final100_avg_coins",
    "train_self_kill_rate",
    "train_final100_self_kill_rate",
    "eval_avg_coins",
    "eval_median_coins",
    "eval_self_kill_rate",
    "eval_avg_steps",
    "completion_rate",
    "timeout_rate",
]


def experiment_paths(task, agent):
    """Return result directory and CSV output paths for one task/agent."""
    output_name = SUPPORTED_AGENTS[agent]

    result_root = Path(f"results/task{task}") / agent
    runs_csv = Path(f"docs/experiments/task{task}_{output_name}_runs.csv")
    summary_csv = Path(f"docs/experiments/task{task}_{output_name}_summary.csv")

    return result_root, runs_csv, summary_csv


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def round_number(name):
    """
    Parse a round number from framework-generated round labels.

    Example:
        'Round 01 (2026-09-11 ...)' -> 1
        'Round 600 (...)' -> 600
    """
    match = re.search(r"Round\s+(\d+)", name)

    if match is None:
        raise ValueError(f"Could not parse round number from: {name}")

    return int(match.group(1))


def extract_rounds(stats):
    """Return sorted round-level statistics needed by Task 1 and Task 2."""
    rounds = []

    for name, data in stats["by_round"].items():
        rounds.append(
            {
                "round": round_number(name),
                "coins": data.get("coins", 0),
                "steps": data.get("steps", 0),
                "suicides": data.get("suicides", 0),
                "kills": data.get("kills", 0),
            }
        )

    return sorted(rounds, key=lambda row: row["round"])


def final_window(rounds, size=FINAL_TRAIN_WINDOW):
    """Return the final training window, or all rounds if fewer are available."""
    return rounds[-size:]


def safe_rate(count, total):
    if total == 0:
        return 0.0
    return count / total


def analyze_task1_run(run_dir):
    """Analyze one Task 1 (coin-heaven) experiment directory."""
    match = TASK1_RUN_PATTERN.match(run_dir.name)

    if match is None:
        return None

    feature_mode, reward_mode, seed = match.groups()

    train_path = run_dir / "train.json"
    eval_path = run_dir / "eval.json"

    if not train_path.exists() or not eval_path.exists():
        print(f"Skipping incomplete run: {run_dir.name}")
        return None

    train_rounds = extract_rounds(load_json(train_path))
    eval_rounds = extract_rounds(load_json(eval_path))

    if not train_rounds or not eval_rounds:
        print(f"Skipping empty run: {run_dir.name}")
        return None

    train_coins = [row["coins"] for row in train_rounds]
    final_train_coins = [row["coins"] for row in final_window(train_rounds)]
    eval_coins = [row["coins"] for row in eval_rounds]

    completed = [
        row
        for row in eval_rounds
        if row["coins"] >= TASK1_TOTAL_COINS
    ]

    timed_out = [
        row
        for row in eval_rounds
        if row["steps"] >= MAX_STEPS
        and row["coins"] < TASK1_TOTAL_COINS
    ]

    completed_steps = [row["steps"] for row in completed]

    return {
        "feature": feature_mode.upper(),
        "reward": reward_mode,
        "seed": int(seed),
        "train_avg_coins": mean(train_coins),
        "train_final100_avg_coins": mean(final_train_coins),
        "eval_avg_coins": mean(eval_coins),
        "eval_median_coins": median(eval_coins),
        "completion_rate": safe_rate(len(completed), len(eval_rounds)),
        "timeout_rate": safe_rate(len(timed_out), len(eval_rounds)),
        "completed_avg_steps": (
            mean(completed_steps)
            if completed_steps else None
        ),
    }


def analyze_task2_run(run_dir):
    """
    Analyze one Task 2 (classic, no opponents) experiment directory.

    Definitions:
      self-kill rate:
        fraction of rounds with at least one suicide.

      completion:
        round ended before MAX_STEPS without a suicide.

        In the no-opponent Task 2 setup, a surviving round that ends before
        the time limit corresponds to the environment's natural completion
        condition rather than death or timeout.

      timeout:
        no suicide and steps >= MAX_STEPS.
    """
    match = TASK2_RUN_PATTERN.match(run_dir.name)

    if match is None:
        return None

    feature_mode, reward_mode, seed = match.groups()

    train_path = run_dir / "train.json"
    eval_path = run_dir / "eval.json"

    if not train_path.exists() or not eval_path.exists():
        print(f"Skipping incomplete run: {run_dir.name}")
        return None

    train_rounds = extract_rounds(load_json(train_path))
    eval_rounds = extract_rounds(load_json(eval_path))

    if not train_rounds or not eval_rounds:
        print(f"Skipping empty run: {run_dir.name}")
        return None

    final_train_rounds = final_window(train_rounds)

    train_coins = [row["coins"] for row in train_rounds]
    final_train_coins = [row["coins"] for row in final_train_rounds]
    eval_coins = [row["coins"] for row in eval_rounds]
    eval_steps = [row["steps"] for row in eval_rounds]

    train_self_kills = sum(row["suicides"] > 0 for row in train_rounds)
    final_train_self_kills = sum(
        row["suicides"] > 0 for row in final_train_rounds
    )
    eval_self_kills = sum(row["suicides"] > 0 for row in eval_rounds)

    completed = [
        row
        for row in eval_rounds
        if row["suicides"] == 0
        and row["steps"] < MAX_STEPS
    ]

    timed_out = [
        row
        for row in eval_rounds
        if row["suicides"] == 0
        and row["steps"] >= MAX_STEPS
    ]

    completed_steps = [row["steps"] for row in completed]

    return {
        "feature": feature_mode.upper(),
        "reward": reward_mode,
        "seed": int(seed),
        "train_avg_coins": mean(train_coins),
        "train_final100_avg_coins": mean(final_train_coins),
        "train_self_kill_rate": safe_rate(
            train_self_kills, len(train_rounds)
        ),
        "train_final100_self_kill_rate": safe_rate(
            final_train_self_kills, len(final_train_rounds)
        ),
        "eval_avg_coins": mean(eval_coins),
        "eval_median_coins": median(eval_coins),
        "eval_self_kill_rate": safe_rate(
            eval_self_kills, len(eval_rounds)
        ),
        "eval_avg_steps": mean(eval_steps),
        "completion_rate": safe_rate(len(completed), len(eval_rounds)),
        "timeout_rate": safe_rate(len(timed_out), len(eval_rounds)),
        "completed_avg_steps": (
            mean(completed_steps)
            if completed_steps else None
        ),
    }


def analyze_run(task, run_dir):
    if task == 1:
        return analyze_task1_run(run_dir)

    if task == 2:
        return analyze_task2_run(run_dir)

    raise ValueError(f"Unsupported task: {task}")


def metrics_for_task(task):
    if task == 1:
        return TASK1_METRICS
    if task == 2:
        return TASK2_METRICS
    raise ValueError(f"Unsupported task: {task}")


def sample_std(values):
    """Return sample standard deviation, or None if fewer than 2 values."""
    return stdev(values) if len(values) >= 2 else None


def aggregate_results(task, results):
    """Aggregate per-seed results by feature/reward configuration."""
    grouped = defaultdict(list)

    for result in results:
        key = (result["feature"], result["reward"])
        grouped[key].append(result)

    summaries = []
    metrics = metrics_for_task(task)

    for (feature, reward), runs in grouped.items():
        row = {
            "feature": feature,
            "reward": reward,
            "n_seeds": len(runs),
            "seeds": ",".join(
                str(run["seed"])
                for run in sorted(runs, key=lambda item: item["seed"])
            ),
        }

        for metric in metrics:
            values = [run[metric] for run in runs]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = sample_std(values)

        completed_steps = [
            run["completed_avg_steps"]
            for run in runs
            if run["completed_avg_steps"] is not None
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
        key=lambda row: (
            row["feature"],
            REWARD_ORDER[row["reward"]],
        )
    )

    return summaries


def format_mean_std(mean_value, std_value, decimals=2, percent=False):
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
    if not rows:
        print()
        print(title)
        print("(no rows)")
        return

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row):
        return " | ".join(
            value.ljust(widths[index])
            for index, value in enumerate(row)
        )

    print()
    print(title)
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))

    for row in rows:
        print(format_row(row))


def print_task1_run_table(results):
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

    for result in results:
        rows.append(
            [
                result["feature"],
                result["reward"],
                str(result["seed"]),
                f'{result["train_avg_coins"]:.2f}',
                f'{result["train_final100_avg_coins"]:.2f}',
                f'{result["eval_avg_coins"]:.2f}',
                f'{result["eval_median_coins"]:.1f}',
                f'{100 * result["completion_rate"]:.1f}%',
                f'{100 * result["timeout_rate"]:.1f}%',
            ]
        )

    print_text_table("Per-seed results", headers, rows)


def print_task2_run_table(results):
    headers = [
        "Feature",
        "Reward",
        "Seed",
        "Train avg",
        "Final 100",
        "Train self-kill",
        "Final100 self-kill",
        "Eval avg",
        "Eval self-kill",
        "Eval steps",
        "Completion",
        "Timeout",
    ]

    rows = []

    for result in results:
        rows.append(
            [
                result["feature"],
                result["reward"],
                str(result["seed"]),
                f'{result["train_avg_coins"]:.2f}',
                f'{result["train_final100_avg_coins"]:.2f}',
                f'{100 * result["train_self_kill_rate"]:.1f}%',
                f'{100 * result["train_final100_self_kill_rate"]:.1f}%',
                f'{result["eval_avg_coins"]:.2f}',
                f'{100 * result["eval_self_kill_rate"]:.1f}%',
                f'{result["eval_avg_steps"]:.1f}',
                f'{100 * result["completion_rate"]:.1f}%',
                f'{100 * result["timeout_rate"]:.1f}%',
            ]
        )

    print_text_table("Per-seed results", headers, rows)


def print_run_table(task, results):
    if task == 1:
        print_task1_run_table(results)
    else:
        print_task2_run_table(results)


def print_task1_summary_table(summaries):
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

    for result in summaries:
        rows.append(
            [
                result["feature"],
                result["reward"],
                str(result["n_seeds"]),
                format_mean_std(
                    result["train_avg_coins_mean"],
                    result["train_avg_coins_std"],
                ),
                format_mean_std(
                    result["train_final100_avg_coins_mean"],
                    result["train_final100_avg_coins_std"],
                ),
                format_mean_std(
                    result["eval_avg_coins_mean"],
                    result["eval_avg_coins_std"],
                ),
                format_mean_std(
                    result["eval_median_coins_mean"],
                    result["eval_median_coins_std"],
                    decimals=1,
                ),
                format_mean_std(
                    result["completion_rate_mean"],
                    result["completion_rate_std"],
                    decimals=1,
                    percent=True,
                ),
                format_mean_std(
                    result["timeout_rate_mean"],
                    result["timeout_rate_std"],
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


def print_task2_summary_table(summaries):
    headers = [
        "Feature",
        "Reward",
        "n",
        "Train avg",
        "Final 100",
        "Final100 self-kill",
        "Eval avg",
        "Eval self-kill",
        "Eval steps",
        "Completion",
        "Timeout",
    ]

    rows = []

    for result in summaries:
        rows.append(
            [
                result["feature"],
                result["reward"],
                str(result["n_seeds"]),
                format_mean_std(
                    result["train_avg_coins_mean"],
                    result["train_avg_coins_std"],
                ),
                format_mean_std(
                    result["train_final100_avg_coins_mean"],
                    result["train_final100_avg_coins_std"],
                ),
                format_mean_std(
                    result["train_final100_self_kill_rate_mean"],
                    result["train_final100_self_kill_rate_std"],
                    decimals=1,
                    percent=True,
                ),
                format_mean_std(
                    result["eval_avg_coins_mean"],
                    result["eval_avg_coins_std"],
                ),
                format_mean_std(
                    result["eval_self_kill_rate_mean"],
                    result["eval_self_kill_rate_std"],
                    decimals=1,
                    percent=True,
                ),
                format_mean_std(
                    result["eval_avg_steps_mean"],
                    result["eval_avg_steps_std"],
                    decimals=1,
                ),
                format_mean_std(
                    result["completion_rate_mean"],
                    result["completion_rate_std"],
                    decimals=1,
                    percent=True,
                ),
                format_mean_std(
                    result["timeout_rate_mean"],
                    result["timeout_rate_std"],
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


def print_summary_table(task, summaries):
    if task == 1:
        print_task1_summary_table(summaries)
    else:
        print_task2_summary_table(summaries)


def run_fieldnames(task):
    common = [
        "feature",
        "reward",
        "seed",
    ]

    if task == 1:
        return common + [
            "train_avg_coins",
            "train_final100_avg_coins",
            "eval_avg_coins",
            "eval_median_coins",
            "completion_rate",
            "timeout_rate",
            "completed_avg_steps",
        ]

    return common + [
        "train_avg_coins",
        "train_final100_avg_coins",
        "train_self_kill_rate",
        "train_final100_self_kill_rate",
        "eval_avg_coins",
        "eval_median_coins",
        "eval_self_kill_rate",
        "eval_avg_steps",
        "completion_rate",
        "timeout_rate",
        "completed_avg_steps",
    ]


def summary_fieldnames(task):
    fields = [
        "feature",
        "reward",
        "n_seeds",
        "seeds",
    ]

    for metric in metrics_for_task(task):
        fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
            ]
        )

    fields.extend(
        [
            "completed_avg_steps_mean",
            "completed_avg_steps_std",
        ]
    )

    return fields


def save_runs_csv(task, results, runs_csv):
    runs_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(runs_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=run_fieldnames(task),
        )
        writer.writeheader()
        writer.writerows(results)


def save_summary_csv(task, summaries, summary_csv):
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary_fieldnames(task),
        )
        writer.writeheader()
        writer.writerows(summaries)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Task 1 or Task 2 Bomberman experiment results."
    )

    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2],
        required=True,
        help="Project task to summarize: 1 or 2.",
    )

    parser.add_argument(
        "--agent",
        choices=sorted(SUPPORTED_AGENTS),
        default="linear_q_agent",
        help="Agent result directory to summarize (default: linear_q_agent).",
    )

    args = parser.parse_args()

    result_root, runs_csv, summary_csv = experiment_paths(
        args.task,
        args.agent,
    )

    if not result_root.exists():
        raise FileNotFoundError(
            f"Result directory not found: {result_root}"
        )

    results = []

    for run_dir in result_root.iterdir():
        if not run_dir.is_dir():
            continue

        result = analyze_run(args.task, run_dir)

        if result is not None:
            results.append(result)

    results.sort(
        key=lambda row: (
            row["feature"],
            REWARD_ORDER[row["reward"]],
            row["seed"],
        )
    )

    if not results:
        print(
            f"No complete Task {args.task} experiment results "
            f"found for {args.agent}."
        )
        return

    summaries = aggregate_results(args.task, results)

    print(f"Task: {args.task}")
    print(f"Agent: {args.agent}")

    print_run_table(args.task, results)
    print_summary_table(args.task, summaries)

    save_runs_csv(args.task, results, runs_csv)
    save_summary_csv(args.task, summaries, summary_csv)

    print()
    print(f"Saved run-level results: {runs_csv}")
    print(f"Saved aggregated summary: {summary_csv}")


if __name__ == "__main__":
    main()
