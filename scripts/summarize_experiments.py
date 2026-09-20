# This utility script was generated with assistance from ChatGPT.
#
# Summarize Bomberman experiment results for Task 1, Task 2, and Task 3.
#
# Examples:
#   python scripts/summarize_experiments.py --task 1 --agent linear_q_agent
#   python scripts/summarize_experiments.py --task 1 --agent sarsa_lambda_agent
#   python scripts/summarize_experiments.py --task 2 --agent linear_q_agent
#   python scripts/summarize_experiments.py --task 2 --agent sarsa_lambda_agent
#   python scripts/summarize_experiments.py --task 3 --agent sarsa_lambda_agent

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

# Task 2 supports F2, F3, F4, and F5 feature configurations.
# Supporting sparse here as well keeps the summarizer useful
# if a sparse control experiment is added later.
TASK2_RUN_PATTERN = re.compile(
    r"^(f[23456])_(sparse|basic|shaped)_seed(\d+)$"
)

# Task 3 experiments use opponents. Keep the feature part flexible so that
# F4 baselines, F5 opponent-aware models, and later feature variants can all
# be summarized without changing this script again.
TASK3_RUN_PATTERN = re.compile(
    r"^(f\d+)_(sparse|basic|shaped)_seed(\d+)$"
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

    "eval_bombs_per_round",
    "eval_crates_per_round",
    "eval_crates_per_bomb",
    "eval_coins_per_100_steps",
    "eval_wait_rate",

    "completion_rate",
    "timeout_rate",
]

# Task 3 has multiple agents, so per-agent lifetime statistics are used for
# coins, kills, suicides, score, actions, bombs, and crates. The framework's
# by_round statistics aggregate all agents, so target-agent final-100 metrics
# cannot be reconstructed reliably from the saved JSON and are intentionally
# omitted here.
TASK3_METRICS = [
    # train.json is optional for Task 3. These values are None when only
    # eval.json is available.
    "train_coins_per_round",
    "train_kills_per_round",
    "train_self_kill_rate",
    "train_score_per_round",
    "train_agent_steps_per_round",

    "eval_coins_per_round",
    "eval_kills_per_round",
    "eval_self_kill_rate",
    "eval_score_per_round",
    "eval_agent_steps_per_round",
    "eval_avg_round_steps",

    "eval_bombs_per_round",
    "eval_crates_per_round",
    "eval_kills_per_bomb",
    "eval_coins_per_100_steps",
    "eval_kills_per_100_steps",
    "eval_wait_rate",
    "timeout_rate",
]

# The game score in the supplied framework uses +1 per coin and +5 per
# opponent kill. This is only used when a Task 3 evaluation is summarized
# directly from a text log rather than from --save-stats JSON.
GAME_COIN_SCORE = 1
GAME_KILL_SCORE = 5

LOG_ROUND_RE = re.compile(r"STARTING ROUND #(\d+)")
LOG_STEP_RE = re.compile(r"STARTING STEP (\d+)")
LOG_ACTION_RE = re.compile(r"Agent <([^>]+)> chose action ([A-Z]+)")
LOG_COIN_RE = re.compile(r"Agent <([^>]+)> picked up coin")
LOG_SELF_KILL_RE = re.compile(r"Agent <([^>]+)> blown up by own bomb")
LOG_KILL_RE = re.compile(r"Agent <([^>]+)> blown up by agent <([^>]+)>'s bomb")
LOG_BOMB_RE = re.compile(r"Agent <([^>]+)> drops bomb")


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


def single_agent_stats(stats):
    """Return lifetime statistics for a single-agent experiment."""
    by_agent = stats.get("by_agent", {})

    if len(by_agent) != 1:
        raise ValueError(
            f"Expected exactly one agent, found {len(by_agent)}."
        )

    return next(iter(by_agent.values()))


def target_agent_stats(stats, agent_name):
    """Return lifetime statistics for the requested agent in a multi-agent run."""
    by_agent = stats.get("by_agent", {})

    if agent_name in by_agent:
        return by_agent[agent_name]

    # The framework appends _0, _1, ... when the same agent code appears
    # multiple times. Accept a unique suffixed match, but reject ambiguity.
    matches = [
        values
        for name, values in by_agent.items()
        if name.startswith(f"{agent_name}_")
    ]

    if len(matches) == 1:
        return matches[0]

    available = ", ".join(sorted(by_agent)) or "(none)"
    raise ValueError(
        f"Could not uniquely identify target agent '{agent_name}'. "
        f"Available agents: {available}"
    )


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

    train_stats = load_json(train_path)
    eval_stats = load_json(eval_path)

    train_rounds = extract_rounds(train_stats)
    eval_rounds = extract_rounds(eval_stats)

    if not train_rounds or not eval_rounds:
        print(f"Skipping empty run: {run_dir.name}")
        return None

    final_train_rounds = final_window(train_rounds)

    train_coins = [row["coins"] for row in train_rounds]
    final_train_coins = [row["coins"] for row in final_train_rounds]
    eval_coins = [row["coins"] for row in eval_rounds]
    eval_steps = [row["steps"] for row in eval_rounds]

    eval_agent_stats = single_agent_stats(eval_stats)

    eval_bombs = eval_agent_stats.get("bombs", 0)
    eval_crates = eval_agent_stats.get("crates", 0)
    eval_moves = eval_agent_stats.get("moves", 0)
    eval_invalid = eval_agent_stats.get("invalid", 0)
    eval_total_steps = eval_agent_stats.get("steps", 0)

    eval_waits = (
        eval_total_steps
        - eval_moves
        - eval_bombs
        - eval_invalid
    )

    if eval_waits < 0:
        raise ValueError(
            f"Derived negative wait count in {run_dir.name}: {eval_waits}"
        )

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
        "train_self_kill_rate": safe_rate( train_self_kills, len(train_rounds)),
        "train_final100_self_kill_rate": safe_rate(final_train_self_kills, len(final_train_rounds)),
        "eval_avg_coins": mean(eval_coins),
        "eval_median_coins": median(eval_coins),
        "eval_self_kill_rate": safe_rate(eval_self_kills, len(eval_rounds)),
        "eval_avg_steps": mean(eval_steps),
        "eval_bombs_per_round": safe_rate(eval_bombs, len(eval_rounds)),
        "eval_crates_per_round": safe_rate(eval_crates, len(eval_rounds)),
        "eval_crates_per_bomb": safe_rate(eval_crates, eval_bombs),
        "eval_coins_per_100_steps": (100.0 * safe_rate(sum(eval_coins), eval_total_steps)),
        "eval_wait_rate": safe_rate(eval_waits, eval_total_steps),
        "completion_rate": safe_rate(len(completed), len(eval_rounds)),
        "timeout_rate": safe_rate(len(timed_out), len(eval_rounds)),
        "completed_avg_steps": (mean(completed_steps) if completed_steps else None),
    }


def _task3_train_metrics(train_stats, agent_name):
    """Return Task 3 training metrics, or None-valued fields when unavailable."""
    empty = {
        "train_coins_per_round": None,
        "train_kills_per_round": None,
        "train_self_kill_rate": None,
        "train_score_per_round": None,
        "train_agent_steps_per_round": None,
    }

    if train_stats is None:
        return empty

    train_rounds = extract_rounds(train_stats)
    if not train_rounds:
        return empty

    train_agent = target_agent_stats(train_stats, agent_name)
    n_train_rounds = len(train_rounds)

    return {
        "train_coins_per_round": safe_rate(train_agent.get("coins", 0), n_train_rounds),
        "train_kills_per_round": safe_rate(train_agent.get("kills", 0), n_train_rounds),
        "train_self_kill_rate": safe_rate(train_agent.get("suicides", 0), n_train_rounds),
        "train_score_per_round": safe_rate(train_agent.get("score", 0), n_train_rounds),
        "train_agent_steps_per_round": safe_rate(train_agent.get("steps", 0), n_train_rounds),
    }


def analyze_task3_run(run_dir, agent_name):
    """
    Analyze one Task 3 multi-agent experiment directory.

    Expected directory name:
        f5_basic_seed123
        f5_shaped_seed456
        ...

    Required file:
        eval.json

    Optional file:
        train.json

    Task 3 uses ``by_agent`` for agent-specific statistics because the
    framework's ``by_round`` section aggregates coins/kills/suicides over all
    agents. Therefore per-agent final-100 training metrics cannot be recovered
    from the standard JSON output.
    """
    match = TASK3_RUN_PATTERN.match(run_dir.name)

    if match is None:
        return None

    feature_mode, reward_mode, seed = match.groups()

    train_path = run_dir / "train.json"
    eval_path = run_dir / "eval.json"

    # For Task 3 an eval-only run is useful and should not be skipped.
    if not eval_path.exists():
        print(f"Skipping Task 3 run without eval.json: {run_dir.name}")
        return None

    train_stats = load_json(train_path) if train_path.exists() else None
    eval_stats = load_json(eval_path)

    eval_rounds = extract_rounds(eval_stats)
    if not eval_rounds:
        print(f"Skipping empty Task 3 evaluation: {run_dir.name}")
        return None

    eval_agent = target_agent_stats(eval_stats, agent_name)
    n_eval_rounds = len(eval_rounds)

    eval_coins = eval_agent.get("coins", 0)
    eval_kills = eval_agent.get("kills", 0)
    eval_suicides = eval_agent.get("suicides", 0)
    eval_score = eval_agent.get("score", 0)
    eval_steps = eval_agent.get("steps", 0)
    eval_bombs = eval_agent.get("bombs", 0)
    eval_crates = eval_agent.get("crates", 0)
    eval_moves = eval_agent.get("moves", 0)
    eval_invalid = eval_agent.get("invalid", 0)

    # Framework lifetime stats do not store WAIT explicitly. Every agent turn
    # is either a successful move, successful bomb, invalid action, or WAIT.
    eval_waits = eval_steps - eval_moves - eval_bombs - eval_invalid

    if eval_waits < 0:
        raise ValueError(
            f"Derived negative wait count in {run_dir.name}: {eval_waits}"
        )

    eval_round_steps = [row["steps"] for row in eval_rounds]
    timed_out = [row for row in eval_rounds if row["steps"] >= MAX_STEPS]

    result = {
        "feature": feature_mode.upper(),
        "reward": reward_mode,
        "seed": int(seed),
        **_task3_train_metrics(train_stats, agent_name),
        "eval_coins_per_round": safe_rate(eval_coins, n_eval_rounds),
        "eval_kills_per_round": safe_rate(eval_kills, n_eval_rounds),
        "eval_self_kill_rate": safe_rate(eval_suicides, n_eval_rounds),
        "eval_score_per_round": safe_rate(eval_score, n_eval_rounds),
        "eval_agent_steps_per_round": safe_rate(eval_steps, n_eval_rounds),
        "eval_avg_round_steps": mean(eval_round_steps),
        "eval_bombs_per_round": safe_rate(eval_bombs, n_eval_rounds),
        "eval_crates_per_round": safe_rate(eval_crates, n_eval_rounds),
        "eval_kills_per_bomb": safe_rate(eval_kills, eval_bombs),
        "eval_coins_per_100_steps": 100.0 * safe_rate(eval_coins, eval_steps),
        "eval_kills_per_100_steps": 100.0 * safe_rate(eval_kills, eval_steps),
        "eval_wait_rate": safe_rate(eval_waits, eval_steps),
        "timeout_rate": safe_rate(len(timed_out), n_eval_rounds),
    }

    return result


def analyze_task3_log(log_path, agent_name):
    """
    Summarize a Task 3 evaluation directly from a framework game.log file.

    This is intended for quick one-off evaluations when --save-stats was not
    used. It can recover actions, WAIT rate, bombs, coins, kills, self-kills,
    deaths by opponents, round lengths, and timeout rate. Crate destruction is
    not logged explicitly, so that metric is unavailable in log mode.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    rounds = {}
    current_round = None
    actions = defaultdict(int)
    coins = 0
    kills = 0
    self_kills = 0
    deaths_by_opponent = 0
    bombs = 0
    timeout_rounds = set()

    with open(log_path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            round_match = LOG_ROUND_RE.search(line)
            if round_match:
                current_round = int(round_match.group(1))
                rounds.setdefault(current_round, 0)
                continue

            step_match = LOG_STEP_RE.search(line)
            if step_match and current_round is not None:
                rounds[current_round] = max(
                    rounds.get(current_round, 0),
                    int(step_match.group(1)),
                )

            action_match = LOG_ACTION_RE.search(line)
            if action_match and action_match.group(1) == agent_name:
                actions[action_match.group(2)] += 1

            coin_match = LOG_COIN_RE.search(line)
            if coin_match and coin_match.group(1) == agent_name:
                coins += 1

            self_kill_match = LOG_SELF_KILL_RE.search(line)
            if self_kill_match and self_kill_match.group(1) == agent_name:
                self_kills += 1

            kill_match = LOG_KILL_RE.search(line)
            if kill_match:
                victim, owner = kill_match.groups()
                if owner == agent_name:
                    kills += 1
                if victim == agent_name and owner != agent_name:
                    deaths_by_opponent += 1

            bomb_match = LOG_BOMB_RE.search(line)
            if bomb_match and bomb_match.group(1) == agent_name:
                bombs += 1

            if "Maximum number of steps reached" in line and current_round is not None:
                timeout_rounds.add(current_round)

    n_rounds = len(rounds)
    total_actions = sum(actions.values())
    waits = actions.get("WAIT", 0)
    total_deaths = self_kills + deaths_by_opponent
    round_steps = list(rounds.values())

    if n_rounds == 0:
        raise ValueError(f"No rounds found in log: {log_path}")

    return {
        "rounds": n_rounds,
        "coins": coins,
        "kills": kills,
        "self_kills": self_kills,
        "deaths_by_opponent": deaths_by_opponent,
        "death_rate": safe_rate(total_deaths, n_rounds),
        "score_per_round": safe_rate(
            GAME_COIN_SCORE * coins + GAME_KILL_SCORE * kills,
            n_rounds,
        ),
        "agent_steps_per_round": safe_rate(total_actions, n_rounds),
        "avg_round_steps": mean(round_steps) if round_steps else 0.0,
        "bombs_per_round": safe_rate(bombs, n_rounds),
        "kills_per_bomb": safe_rate(kills, bombs),
        "coins_per_100_steps": 100.0 * safe_rate(coins, total_actions),
        "kills_per_100_steps": 100.0 * safe_rate(kills, total_actions),
        "wait_rate": safe_rate(waits, total_actions),
        "timeout_rate": safe_rate(len(timeout_rounds), n_rounds),
        "actions": dict(actions),
    }


def print_task3_log_summary(log_path, agent_name):
    result = analyze_task3_log(log_path, agent_name)

    print("Task 3 log evaluation")
    print(f"Agent: {agent_name}")
    print(f"Log: {log_path}")
    print(f"Rounds: {result['rounds']}")
    print(f"Coins: {result['coins']} ({result['coins'] / result['rounds']:.2f}/round)")
    print(f"Kills: {result['kills']} ({result['kills'] / result['rounds']:.2f}/round)")
    print(f"Self-kills: {result['self_kills']}")
    print(f"Deaths by opponents: {result['deaths_by_opponent']}")
    print(f"Death rate: {100 * result['death_rate']:.1f}%")
    print(f"Score/round: {result['score_per_round']:.2f}")
    print(f"Agent steps/round: {result['agent_steps_per_round']:.1f}")
    print(f"Average round steps: {result['avg_round_steps']:.1f}")
    print(f"Bombs/round: {result['bombs_per_round']:.2f}")
    print(f"Kills/bomb: {result['kills_per_bomb']:.3f}")
    print(f"Coins/100 agent steps: {result['coins_per_100_steps']:.2f}")
    print(f"Kills/100 agent steps: {result['kills_per_100_steps']:.2f}")
    print(f"WAIT rate: {100 * result['wait_rate']:.1f}%")
    print(f"Timeout rate: {100 * result['timeout_rate']:.1f}%")
    print("Actions: " + ", ".join(
        f"{action}={count}" for action, count in sorted(result["actions"].items())
    ))

def analyze_run(task, run_dir, agent_name):
    if task == 1:
        return analyze_task1_run(run_dir)

    if task == 2:
        return analyze_task2_run(run_dir)

    if task == 3:
        return analyze_task3_run(run_dir, agent_name)

    raise ValueError(f"Unsupported task: {task}")


def metrics_for_task(task):
    if task == 1:
        return TASK1_METRICS
    if task == 2:
        return TASK2_METRICS
    if task == 3:
        return TASK3_METRICS
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
            # Task 3 may be eval-only, in which case train metrics are None.
            values = [run[metric] for run in runs if run[metric] is not None]
            row[f"{metric}_mean"] = mean(values) if values else None
            row[f"{metric}_std"] = sample_std(values) if values else None

        if task in {1, 2}:
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
        "Bombs/r",
        "Crates/r",
        "Crates/bomb",
        "Coins/100",
        "Wait rate",
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
                f'{result["eval_bombs_per_round"]:.2f}',
                f'{result["eval_crates_per_round"]:.2f}',
                f'{result["eval_crates_per_bomb"]:.2f}',
                f'{result["eval_coins_per_100_steps"]:.2f}',
                f'{100 * result["eval_wait_rate"]:.1f}%',
            ]
        )

    print_text_table("Per-seed results", headers, rows)


def _fmt(value, decimals=2, percent=False):
    if value is None:
        return "-"
    if percent:
        return f"{100 * value:.{decimals}f}%"
    return f"{value:.{decimals}f}"


def print_task3_run_table(results):
    headers = [
        "Feature",
        "Reward",
        "Seed",
        "Train coin/r",
        "Train kill/r",
        "Train self-kill",
        "Eval coin/r",
        "Eval kill/r",
        "Eval self-kill",
        "Eval score/r",
        "Agent steps/r",
        "Timeout",
        "Bombs/r",
        "Crates/r",
        "Kills/bomb",
        "Coins/100",
        "Kills/100",
        "Wait rate",
    ]

    rows = []

    for result in results:
        rows.append(
            [
                result["feature"],
                result["reward"],
                str(result["seed"]),
                _fmt(result["train_coins_per_round"]),
                _fmt(result["train_kills_per_round"]),
                _fmt(result["train_self_kill_rate"], decimals=1, percent=True),
                _fmt(result["eval_coins_per_round"]),
                _fmt(result["eval_kills_per_round"]),
                _fmt(result["eval_self_kill_rate"], decimals=1, percent=True),
                _fmt(result["eval_score_per_round"]),
                _fmt(result["eval_agent_steps_per_round"], decimals=1),
                _fmt(result["timeout_rate"], decimals=1, percent=True),
                _fmt(result["eval_bombs_per_round"]),
                _fmt(result["eval_crates_per_round"]),
                _fmt(result["eval_kills_per_bomb"], decimals=3),
                _fmt(result["eval_coins_per_100_steps"]),
                _fmt(result["eval_kills_per_100_steps"]),
                _fmt(result["eval_wait_rate"], decimals=1, percent=True),
            ]
        )

    print_text_table("Per-seed results", headers, rows)

def print_run_table(task, results):
    if task == 1:
        print_task1_run_table(results)
    elif task == 2:
        print_task2_run_table(results)
    else:
        print_task3_run_table(results)


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
        "Bombs/r",
        "Crates/r",
        "Crates/bomb",
        "Coins/100",
        "Wait rate",
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
                format_mean_std(
                    result["eval_bombs_per_round_mean"],
                    result["eval_bombs_per_round_std"],
                ),
                format_mean_std(
                    result["eval_crates_per_round_mean"],
                    result["eval_crates_per_round_std"],
                ),
                format_mean_std(
                    result["eval_crates_per_bomb_mean"],
                    result["eval_crates_per_bomb_std"],
                ),
                format_mean_std(
                    result["eval_coins_per_100_steps_mean"],
                    result["eval_coins_per_100_steps_std"],
                ),
                format_mean_std(
                    result["eval_wait_rate_mean"],
                    result["eval_wait_rate_std"],
                    decimals=1,
                    percent=True
                ),
            ]
        )

    print_text_table(
        "Aggregated results (mean +/- sample std across seeds)",
        headers,
        rows,
    )


def print_task3_summary_table(summaries):
    headers = [
        "Feature",
        "Reward",
        "n",
        "Train coin/r",
        "Train kill/r",
        "Train self-kill",
        "Eval coin/r",
        "Eval kill/r",
        "Eval self-kill",
        "Eval score/r",
        "Agent steps/r",
        "Timeout",
        "Bombs/r",
        "Kills/bomb",
        "Coins/100",
        "Kills/100",
        "Wait rate",
    ]

    rows = []

    for result in summaries:
        rows.append(
            [
                result["feature"],
                result["reward"],
                str(result["n_seeds"]),
                format_mean_std(result["train_coins_per_round_mean"], result["train_coins_per_round_std"]),
                format_mean_std(result["train_kills_per_round_mean"], result["train_kills_per_round_std"]),
                format_mean_std(result["train_self_kill_rate_mean"], result["train_self_kill_rate_std"], decimals=1, percent=True),
                format_mean_std(result["eval_coins_per_round_mean"], result["eval_coins_per_round_std"]),
                format_mean_std(result["eval_kills_per_round_mean"], result["eval_kills_per_round_std"]),
                format_mean_std(result["eval_self_kill_rate_mean"], result["eval_self_kill_rate_std"], decimals=1, percent=True),
                format_mean_std(result["eval_score_per_round_mean"], result["eval_score_per_round_std"]),
                format_mean_std(result["eval_agent_steps_per_round_mean"], result["eval_agent_steps_per_round_std"], decimals=1),
                format_mean_std(result["timeout_rate_mean"], result["timeout_rate_std"], decimals=1, percent=True),
                format_mean_std(result["eval_bombs_per_round_mean"], result["eval_bombs_per_round_std"]),
                format_mean_std(result["eval_kills_per_bomb_mean"], result["eval_kills_per_bomb_std"], decimals=3),
                format_mean_std(result["eval_coins_per_100_steps_mean"], result["eval_coins_per_100_steps_std"]),
                format_mean_std(result["eval_kills_per_100_steps_mean"], result["eval_kills_per_100_steps_std"]),
                format_mean_std(result["eval_wait_rate_mean"], result["eval_wait_rate_std"], decimals=1, percent=True),
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
    elif task == 2:
        print_task2_summary_table(summaries)
    else:
        print_task3_summary_table(summaries)


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

    if task == 3:
        return common + TASK3_METRICS

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
        "eval_bombs_per_round",
        "eval_crates_per_round",
        "eval_crates_per_bomb",
        "eval_coins_per_100_steps",
        "eval_wait_rate",
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

    if task in {1, 2}:
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
        description="Summarize Task 1, Task 2, or Task 3 Bomberman experiment results."
    )

    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Project task to summarize: 1, 2, or 3.",
    )

    parser.add_argument(
        "--agent",
        choices=sorted(SUPPORTED_AGENTS),
        default="linear_q_agent",
        help="Agent result directory to summarize (default: linear_q_agent).",
    )

    parser.add_argument(
        "--eval-log",
        type=Path,
        default=None,
        help=(
            "Task 3 only: summarize one evaluation game.log directly. "
            "Useful when --save-stats was not enabled."
        ),
    )

    args = parser.parse_args()

    if args.eval_log is not None:
        if args.task != 3:
            parser.error("--eval-log is only supported with --task 3")
        print_task3_log_summary(args.eval_log, args.agent)
        return

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

        result = analyze_run(args.task, run_dir, args.agent)

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
