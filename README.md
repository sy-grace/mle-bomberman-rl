# Reinforcement Learning for Bomberman

This repository contains two feature-based reinforcement-learning agents for the Bomberman framework:

- **Linear Q-learning**
- **Linear SARSA($\lambda$)**

Both agents use a linear action-value function with hand-designed features. During development, the state representation was extended from simple coin navigation to bomb safety, escape behavior, movement stabilization, and opponent-aware play. The final agents also combine learned action values with deterministic safety and action-filtering rules.

## Repository structure

```text
.
├── agent_code/
│   ├── linear_q_agent/
│   ├── sarsa_lambda_agent/
│   ├── coin_collector_agent/
│   ├── peaceful_agent/
│   ├── random_agent/
│   └── rule_based_agent/
├── docs/
│   └── experiments/          # experiment runs and summaries
├── results/                  # generated JSON statistics
├── replays/
├── logs/
├── main.py                   # game / training entry point
├── settings.py               # scenarios and game settings
├── environment.py
└── Dockerfile
```

The final checkpoints are stored as:

```text
agent_code/linear_q_agent/my-saved-model.pt
agent_code/sarsa_lambda_agent/my-saved-model.pt
```

## Final agents

### Linear Q-learning

The final Linear-Q agent uses the **F7** configuration with 41 features.

```text
F0 (7)
→ F1 (11)
→ F2 (25)
→ F3 (26)
→ F4 (27)
→ F5 (31)
→ F6-initial (32)
→ F6-refined (32)
→ F7 (41)
```

Later versions added safe bomb placement, crate-yield information, movement stabilization, persistent escape handling, action filtering, and opponent-aware combat features.

### SARSA($\lambda$)

The final SARSA agent uses **F7** with 38 features.

```text
F0 (7)
→ F1 (11)
→ F2 (25)
→ F3 (26)
→ F4 (27)
├─ F5-Task2 (31)                  [tested, not retained]
└─ F5-Task3 (38)
   → F6 / F6.1 (38)
   → F7 (38)                      [selected]
   → F8 (42)                      [tested, not retained]
```

The final SARSA checkpoint is the **T4.3 F7 control checkpoint (`control123`)**, selected in the held-out T4.7 comparison.

---

## Setup

### Recommended: Docker

```bash
git clone https://github.com/sy-grace/mle-bomberman-rl.git
cd mle-bomberman-rl

docker build -t mle-bomberman .
docker run --rm -it \
  -v "$(pwd):/home/bomberman" \
  -w /home/bomberman \
  mle-bomberman bash
```

Windows PowerShell:

```powershell
docker build -t mle-bomberman .
docker run --rm -it `
  -v "${PWD}:/home/bomberman" `
  -w /home/bomberman `
  mle-bomberman bash
```

The supplied `Dockerfile` defines the reference environment used by the project.

Local execution is also possible if the framework dependencies are installed:

```bash
python main.py --help
```

---

## Run the final checkpoints

Both final agents use `FEATURE_MODE=f7`. Evaluation loads the stored checkpoint and does not train.

Linux / macOS:

```bash
export FEATURE_MODE=f7
export MODEL_START_MODE=resume
```

Windows PowerShell:

```powershell
$env:FEATURE_MODE="f7"
$env:MODEL_START_MODE="resume"
```

### SARSA vs. three rule-based agents

```bash
python main.py play \
  --agents sarsa_lambda_agent rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic \
  --seed 4004 \
  --n-rounds 100 \
  --no-gui \
  --save-stats results/final_sarsa_4004.json
```

### Linear-Q vs. three rule-based agents

```bash
python main.py play \
  --agents linear_q_agent rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic \
  --seed 4004 \
  --n-rounds 100 \
  --no-gui \
  --save-stats results/final_linear_4004.json
```

Repeat with seeds:

```text
4004, 5005, 6006
```

These are the held-out seeds used in the final benchmark.

---

## Reproduce the final direct comparison

The final direct comparison used both learned agents together with two rule-based agents. Two slot orders were evaluated for every seed.

For each of `4004`, `5005`, and `6006`, run:

```bash
python main.py play \
  --agents sarsa_lambda_agent linear_q_agent rule_based_agent rule_based_agent \
  --scenario classic \
  --seed 4004 \
  --n-rounds 100 \
  --no-gui \
  --save-stats results/duel_sarsa_first_4004.json
```

and:

```bash
python main.py play \
  --agents linear_q_agent sarsa_lambda_agent rule_based_agent rule_based_agent \
  --scenario classic \
  --seed 4004 \
  --n-rounds 100 \
  --no-gui \
  --save-stats results/duel_linear_first_4004.json
```

This gives:

```text
3 seeds × 2 slot orders × 100 rounds = 600 rounds
```

The recorded final results are stored in:

```text
docs/experiments/task4_final_model_selection_runs.csv
docs/experiments/task4_final_model_selection_summary.csv
docs/experiments/task4_final_model_selection_head_to_head.csv
```

### Recorded final benchmark

Against three rule-based opponents:

| Agent | Score / round | First place | Survival | Self-kill | Kills / round |
|---|---:|---:|---:|---:|---:|
| Linear-Q | 4.49 | 42.3% | 38.0% | 38.7% | 0.410 |
| SARSA | 4.81 | 46.0% | 70.7% | 16.3% | 0.343 |

Combined direct comparison over 600 rounds:

| Metric | Linear-Q | SARSA |
|---|---:|---:|
| Higher-score rate | 35.3% | 58.7% |
| First-place rate | 27.8% | 44.3% |
| Mean rank | 2.52 | 2.05 |
| Survival rate | 25.8% | 54.0% |

Mean SARSA minus Linear-Q score difference: **+1.32 points per round**.  
Equal-score rate: **6.0%**.

These values compare the two final agents as complete systems. They are not a controlled comparison of only the Q-learning and SARSA update rules because the final agents also differ in features, rewards, and safety logic.

---

## Configuration

The agents use environment variables for experiment configuration.

| Variable | Values | Purpose |
|---|---|---|
| `FEATURE_MODE` | Linear-Q: `f0`...`f7`; SARSA: `f0`...`f8` | feature / policy configuration |
| `REWARD_MODE` | `sparse`, `basic`, `shaped`; SARSA also supports `hunt_extra` | reward configuration |
| `MODEL_START_MODE` | `fresh`, `resume` | initialize or load a checkpoint |
| `EXPERIMENT_SEED` | integer | model initialization and exploration RNG |
| `SARSA_LAMBDA` | float, default `0.8` | SARSA trace parameter |

The framework option `--seed` controls the Bomberman environment RNG. For reproducible training, set both `EXPERIMENT_SEED` and `--seed`.

Common learning parameters:

```text
alpha            = 0.01
gamma            = 0.9
epsilon start    = 1.0
epsilon decay    = 0.99 per round
epsilon minimum  = 0.01
SARSA lambda     = 0.8
```

---

## Training

> **Important:** training overwrites `agent_code/<agent>/my-saved-model.pt`. Use a fresh clone or back up the final checkpoints before reproducing training runs.

Example fresh Linear-Q training:

```bash
MODEL_START_MODE=fresh \
FEATURE_MODE=f1 \
REWARD_MODE=basic \
EXPERIMENT_SEED=123 \
python main.py play \
  --agents linear_q_agent \
  --train 1 \
  --scenario coin-heaven \
  --seed 123 \
  --n-rounds 600 \
  --no-gui
```

Example fresh SARSA training:

```bash
MODEL_START_MODE=fresh \
FEATURE_MODE=f1 \
REWARD_MODE=basic \
SARSA_LAMBDA=0.8 \
EXPERIMENT_SEED=123 \
python main.py play \
  --agents sarsa_lambda_agent \
  --train 1 \
  --scenario coin-heaven \
  --seed 123 \
  --n-rounds 600 \
  --no-gui
```

Evaluate the trained checkpoint greedily:

```bash
python main.py play \
  --agents sarsa_lambda_agent \
  --scenario coin-heaven \
  --seed 999 \
  --n-rounds 100 \
  --no-gui \
  --save-stats results/example_eval.json
```

Exploration is active only in training mode.

---

## Experimental protocol

### Tasks 1 and 2

Main protocol:

```text
training rounds: 600
training seeds:  123, 456, 2026
evaluation:      100 greedy rounds per trained model
evaluation seed: 999
```

Task 1 used `coin-heaven` and compared F0/F1 with sparse, basic, and shaped rewards. SARSA additionally compared `lambda=0.0` and `lambda=0.8`.

Task 2 used `classic` without opponents and developed bomb-safety and escape features. Linear-Q progressed through F2-F6; SARSA progressed through F2-F5-Task2.

Recorded files:

```text
docs/experiments/task1_*.csv
docs/experiments/task2_*.csv
```

### Task 3

Scenario: `classic`

Opponent setting:

```text
learning_agent + peaceful_agent + coin_collector_agent
```

Typical command:

```bash
python main.py play \
  --agents sarsa_lambda_agent peaceful_agent coin_collector_agent \
  --train 1 \
  --scenario classic \
  --n-rounds 600 \
  --no-gui
```

SARSA development added opponent-aware features, persistent own-bomb escape handling, opponent-bomb safety, immediate-destination safety, and F7 hunt behavior.

Main SARSA Task 3 training seeds:

```text
123, 456, 2026
```

Recorded files:

```text
docs/experiments/task3_sarsa_lambda_runs.csv
docs/experiments/task3_sarsa_lambda_summary.csv
```

### Task 4

Task 4 focused on competitive play against `rule_based_agent`.

| Label | Purpose |
|---|---|
| T4.0 | F7 zero-shot evaluation |
| T4.1 | F7 fine-tuning vs. one rule-based opponent |
| T4.2 | mixed-opponent fine-tuning |
| T4.3 | F7 control vs. F8 tactical features |
| T4.4 | hunt-reward modification |
| T4.5 | multi-opponent stress test |
| T4.6 | fine-tuning vs. three rule-based opponents |
| T4.7 | held-out checkpoint selection |

T4.7 compared:

```text
control123
finetuned456
```

using seeds `1001`, `2002`, and `3003` under three opponent settings. Each candidate was evaluated for 900 rounds in total.

`control123` was selected as the final SARSA checkpoint.

Recorded files:

```text
docs/experiments/task4_sarsa_lambda_runs.csv
docs/experiments/task4_sarsa_lambda_summary.csv
```

---

## Historical experiments

The current `master` branch represents the **final agents**. It does not provide a runtime switch for every historical implementation.

For example:

- Linear-Q `F6-initial` and `F6-refined` share the same feature dimension but differ in later controller / action-filtering logic.
- SARSA F6/F6.1 keep the same feature representation while safety logic changes.
- The label `F5` was reused during development for different representations.

For this reason:

1. Use `docs/experiments/` as the recorded source for historical ablation results.
2. Use current `master` to reproduce the final agents and final benchmark.
3. For an exact historical implementation, check out the corresponding Git commit / pull request.

Useful PRs:

- [#118 – Linear-Q F6 refinement](https://github.com/sy-grace/mle-bomberman-rl/pull/118)
- [#119 – Linear-Q Task 3 opponent hunting](https://github.com/sy-grace/mle-bomberman-rl/pull/119)
- [#130 – SARSA persistent own-bomb escape](https://github.com/sy-grace/mle-bomberman-rl/pull/130)
- [#132 – SARSA opponent-bomb / immediate-destination safety](https://github.com/sy-grace/mle-bomberman-rl/pull/132)
- [#138 – SARSA Task 3 opponent hunting](https://github.com/sy-grace/mle-bomberman-rl/pull/138)
- [#149 – Linear-Q Task 4](https://github.com/sy-grace/mle-bomberman-rl/pull/149)

---

## Tests

Run from the repository root:

```bash
python -m unittest agent_code.linear_q_agent.tests
python -m unittest agent_code.sarsa_lambda_agent.tests
```

The tests cover feature construction, learning updates, reward behavior, and later safety/controller logic.

---

## Saving statistics and replays

Save statistics:

```bash
python main.py play \
  --agents sarsa_lambda_agent rule_based_agent \
  --scenario classic \
  --seed 1001 \
  --n-rounds 100 \
  --no-gui \
  --save-stats results/example.json
```

Save a replay:

```bash
python main.py play \
  --agents sarsa_lambda_agent rule_based_agent \
  --scenario classic \
  --seed 1001 \
  --n-rounds 1 \
  --save-replay
```

Available scenarios:

```text
empty
coin-heaven
loot-crate
classic
```

`classic` is the tournament-style setting.

---

## Development approach

The project followed an iterative workflow:

1. start with a small representation,
2. evaluate it,
3. inspect failure modes,
4. add a targeted feature, reward, or safety mechanism,
5. compare against the previous relevant configuration,
6. retain or reject the change based on evaluation.

Examples include BFS navigation for blocked coin paths, pre-bomb escape checks, persistent escape handling, anti-oscillation logic, immediate-death filtering, and opponent hunt behavior.

---

## Contributors

**Siyeon Kim**
- most Linear-Q experimental development in Tasks 1 and 2
- complete SARSA($\lambda$) branch and Tasks 1-4 SARSA experiments
- Task 4 SARSA model selection
- final cross-agent comparison
- integration and testing

**Binyam Olango**
- initial Linear Q-learning implementation
- Linear-Q F6 refinement
- opponent-aware Linear-Q F7 development
- Linear-Q Task 3 / Task 4 development
- integration and testing

The project was developed collaboratively using separate branches, pull requests, peer review, GitHub Issues, and a Kanban board.

## Notes

- The repository contains the final code, checkpoints, and recorded experiment CSVs.
- Historical ablations may require an earlier commit.
- Training overwrites the active checkpoint.
- For reproducibility, record both `EXPERIMENT_SEED` and `--seed`.
- The final report is intentionally not stored in this public repository.

## AI Assistance

This README was drafted with assistance from OpenAI ChatGPT based on the project's source code, experiment records, GitHub issue and pull-request history, and final report. The authors reviewed and edited the content and are responsible for its accuracy.