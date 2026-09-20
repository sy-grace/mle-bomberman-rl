@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM Task 4.2 mixed-opponent fine-tuning runner
REM
REM Usage:
REM   run_task4_mixed.bat <agent> <feature_mode> <reward_mode> <experiment_seed> [run|eval]
REM
REM Modes:
REM   run  = resume the matching T4.1 fine-tuned checkpoint,
REM          fine-tune one shared model with interleaved opponents,
REM          then evaluate separately against all three opponents. (default)
REM   eval = evaluate an existing T4.2 mixed checkpoint only.
REM
REM Training schedule per seed:
REM   20 rounds vs rule_based_agent
REM   10 rounds vs coin_collector_agent
REM   10 rounds vs peaceful_agent
REM   repeated for 15 cycles = 600 additional rounds total
REM
REM Opponent totals:
REM   rule_based_agent      300 rounds (50%%)
REM   coin_collector_agent  150 rounds (25%%)
REM   peaceful_agent        150 rounds (25%%)
REM
REM Primary evaluation remains rule_based_agent so eval.json stays
REM directly comparable with T4.0/T4.1. Additional opponent evaluations
REM are stored separately.
REM ============================================================

set "AGENT=%~1"
set "FEATURE_MODE=%~2"
set "REWARD_MODE=%~3"
set "BASE_SEED=%~4"
set "RUN_MODE=%~5"

if "%RUN_MODE%"=="" set "RUN_MODE=run"

set "CYCLES=15"
set "RULE_ROUNDS=20"
set "COIN_ROUNDS=10"
set "PEACEFUL_ROUNDS=10"
set "EVAL_ROUNDS=100"
set "EVAL_SEED=999"

set "RULE_OPPONENT=rule_based_agent"
set "COIN_OPPONENT=coin_collector_agent"
set "PEACEFUL_OPPONENT=peaceful_agent"

REM ------------------------------------------------------------
REM Validate arguments
REM ------------------------------------------------------------

if "%AGENT%"=="" (
    echo ERROR: AGENT is missing.
    echo Usage: run_task4_mixed.bat ^<agent^> ^<feature_mode^> ^<basic^|shaped^> ^<seed^> [run^|eval]
    exit /b 1
)

if "%FEATURE_MODE%"=="" (
    echo ERROR: FEATURE_MODE is missing.
    exit /b 1
)

if "%REWARD_MODE%"=="" (
    echo ERROR: REWARD_MODE is missing.
    exit /b 1
)

if "%BASE_SEED%"=="" (
    echo ERROR: EXPERIMENT_SEED is missing.
    exit /b 1
)

if /I not "%REWARD_MODE%"=="basic" if /I not "%REWARD_MODE%"=="shaped" (
    echo ERROR: REWARD_MODE must be basic or shaped for Task 4 experiments.
    exit /b 1
)

if /I not "%RUN_MODE%"=="run" if /I not "%RUN_MODE%"=="eval" (
    echo ERROR: MODE must be run or eval.
    exit /b 1
)

set "SOURCE_DIR=results\task4\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%BASE_SEED%_finetuned"
set "RESULT_DIR=results\task4\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%BASE_SEED%_mixed"
set "ACTIVE_MODEL=agent_code\%AGENT%\my-saved-model.pt"

set "TRAIN_STATS_DIR=%RESULT_DIR%\train_blocks"
set "TRAIN_LOG_ROOT=%RESULT_DIR%\train_logs"
set "EVAL_LOG_DIR=%RESULT_DIR%\eval_logs"
set "EVAL_COIN_LOG_DIR=%RESULT_DIR%\eval_coin_collector_logs"
set "EVAL_PEACEFUL_LOG_DIR=%RESULT_DIR%\eval_peaceful_logs"

if /I "%RUN_MODE%"=="eval" goto EVALUATION_ONLY

REM ============================================================
REM FULL T4.2 RUN: T4.1 CHECKPOINT -> MIXED FINE-TUNING -> EVAL
REM ============================================================

if not exist "%SOURCE_DIR%\model.pt" (
    echo ERROR: Matching T4.1 fine-tuned checkpoint was not found:
    echo   %SOURCE_DIR%\model.pt
    echo Run the corresponding Task 4 finetune experiment first.
    exit /b 1
)

REM Start clean so old block statistics/logs cannot be mixed with this run.
if exist "%RESULT_DIR%" (
    echo Removing existing T4.2 result directory:
    echo   %RESULT_DIR%
    rmdir /S /Q "%RESULT_DIR%"
    if exist "%RESULT_DIR%" (
        echo ERROR: Could not remove existing T4.2 result directory.
        exit /b 1
    )
)

mkdir "%RESULT_DIR%" || exit /b 1
mkdir "%TRAIN_STATS_DIR%" || exit /b 1
mkdir "%TRAIN_LOG_ROOT%" || exit /b 1
mkdir "%EVAL_LOG_DIR%" || exit /b 1
mkdir "%EVAL_COIN_LOG_DIR%" || exit /b 1
mkdir "%EVAL_PEACEFUL_LOG_DIR%" || exit /b 1

REM Preserve the exact T4.1 initialization checkpoint.
copy /Y "%SOURCE_DIR%\model.pt" "%RESULT_DIR%\initial_model.pt" >nul
if errorlevel 1 (
    echo ERROR: Could not preserve the T4.1 initialization checkpoint.
    exit /b 1
)

REM Activate the T4.1 checkpoint for the first mixed-training block.
copy /Y "%SOURCE_DIR%\model.pt" "%ACTIVE_MODEL%" >nul
if errorlevel 1 (
    echo ERROR: Could not activate the T4.1 checkpoint.
    exit /b 1
)

REM Keep a recoverable copy of the current model in the result directory.
copy /Y "%SOURCE_DIR%\model.pt" "%RESULT_DIR%\model.pt" >nul
if errorlevel 1 (
    echo ERROR: Could not initialize the T4.2 checkpoint copy.
    exit /b 1
)

set "MODEL_START_MODE=resume"
set "BLOCK_COUNTER=0"

set /a TOTAL_TRAIN_ROUNDS=CYCLES*(RULE_ROUNDS+COIN_ROUNDS+PEACEFUL_ROUNDS)

echo.
echo ============================================================
echo Task 4.2 - mixed-opponent fine-tuning
echo ============================================================
echo Agent:             %AGENT%
echo Feature mode:      %FEATURE_MODE%
echo Reward mode:       %REWARD_MODE%
echo Experiment seed:   %BASE_SEED%
echo Source checkpoint: %SOURCE_DIR%\model.pt
echo Cycles:            %CYCLES%
echo Schedule/cycle:    %RULE_ROUNDS% rule + %COIN_ROUNDS% coin + %PEACEFUL_ROUNDS% peaceful
echo Training rounds:   %TOTAL_TRAIN_ROUNDS%
echo Evaluation rounds: %EVAL_ROUNDS% per opponent
echo Evaluation seed:   %EVAL_SEED%
echo Result directory:  %RESULT_DIR%
echo ============================================================
echo.

for /L %%C in (1,1,%CYCLES%) do (
    echo.
    echo ============================================================
    echo Mixed-training cycle %%C / %CYCLES%
    echo ============================================================

    call :TRAIN_BLOCK %%C "%RULE_OPPONENT%" %RULE_ROUNDS% rule_based
    if errorlevel 1 exit /b 1

    call :TRAIN_BLOCK %%C "%COIN_OPPONENT%" %COIN_ROUNDS% coin_collector
    if errorlevel 1 exit /b 1

    call :TRAIN_BLOCK %%C "%PEACEFUL_OPPONENT%" %PEACEFUL_ROUNDS% peaceful
    if errorlevel 1 exit /b 1
)

REM Preserve the final mixed checkpoint explicitly.
if not exist "%ACTIVE_MODEL%" (
    echo ERROR: Mixed training finished but the active checkpoint is missing:
    echo   %ACTIVE_MODEL%
    exit /b 1
)

copy /Y "%ACTIVE_MODEL%" "%RESULT_DIR%\model.pt" >nul
if errorlevel 1 (
    echo ERROR: Could not preserve the final T4.2 checkpoint.
    exit /b 1
)

echo.
echo Mixed-opponent fine-tuning complete.
echo Starting separate evaluations...
echo.

goto RUN_EVALUATIONS


REM ============================================================
REM EVALUATION ONLY
REM ============================================================

:EVALUATION_ONLY

if not exist "%RESULT_DIR%\model.pt" (
    echo ERROR: T4.2 mixed checkpoint was not found:
    echo   %RESULT_DIR%\model.pt
    echo Run this script in "run" mode first.
    exit /b 1
)

if not exist "%EVAL_LOG_DIR%" mkdir "%EVAL_LOG_DIR%"
if not exist "%EVAL_COIN_LOG_DIR%" mkdir "%EVAL_COIN_LOG_DIR%"
if not exist "%EVAL_PEACEFUL_LOG_DIR%" mkdir "%EVAL_PEACEFUL_LOG_DIR%"

copy /Y "%RESULT_DIR%\model.pt" "%ACTIVE_MODEL%" >nul
if errorlevel 1 (
    echo ERROR: Could not activate the T4.2 checkpoint for evaluation.
    exit /b 1
)

set "MODEL_START_MODE=resume"

goto RUN_EVALUATIONS


REM ============================================================
REM EVALUATE FINAL T4.2 CHECKPOINT AGAINST EACH OPPONENT
REM ============================================================

:RUN_EVALUATIONS

REM Evaluation is greedy, but set the environment seed explicitly for
REM reproducibility and to match the fixed Task 4 evaluation convention.
set "EXPERIMENT_SEED=%EVAL_SEED%"

call :EVAL_OPPONENT "%RULE_OPPONENT%" "%RESULT_DIR%\eval.json" "%EVAL_LOG_DIR%" rule_based
if errorlevel 1 exit /b 1

call :EVAL_OPPONENT "%COIN_OPPONENT%" "%RESULT_DIR%\eval_coin_collector.json" "%EVAL_COIN_LOG_DIR%" coin_collector
if errorlevel 1 exit /b 1

call :EVAL_OPPONENT "%PEACEFUL_OPPONENT%" "%RESULT_DIR%\eval_peaceful.json" "%EVAL_PEACEFUL_LOG_DIR%" peaceful
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo Task 4.2 mixed-opponent experiment finished successfully.
echo ============================================================
echo Initial T4.1 checkpoint:
echo   %RESULT_DIR%\initial_model.pt
echo Final T4.2 checkpoint:
echo   %RESULT_DIR%\model.pt
echo Primary rule-based evaluation:
echo   %RESULT_DIR%\eval.json
echo Coin-collector evaluation:
echo   %RESULT_DIR%\eval_coin_collector.json
echo Peaceful-agent evaluation:
echo   %RESULT_DIR%\eval_peaceful.json
echo ============================================================
echo.

endlocal
exit /b 0


REM ============================================================
REM SUBROUTINE: ONE TRAINING BLOCK
REM
REM Args:
REM   %1 = cycle number
REM   %2 = opponent name
REM   %3 = number of rounds
REM   %4 = short label
REM ============================================================

:TRAIN_BLOCK
set "CYCLE_NUMBER=%~1"
set "BLOCK_OPPONENT=%~2"
set "BLOCK_ROUNDS=%~3"
set "BLOCK_LABEL=%~4"

set /a BLOCK_COUNTER+=1
set /a BLOCK_SEED=BASE_SEED*1000+BLOCK_COUNTER

REM The framework seed and the agent's EXPERIMENT_SEED are both varied
REM across blocks so restarting Python does not replay the same RNG stream.
set "EXPERIMENT_SEED=!BLOCK_SEED!"

set "PADDED_CYCLE=0!CYCLE_NUMBER!"
set "PADDED_CYCLE=!PADDED_CYCLE:~-2!"
set "PADDED_BLOCK=0!BLOCK_COUNTER!"
set "PADDED_BLOCK=!PADDED_BLOCK:~-2!"

set "BLOCK_NAME=cycle!PADDED_CYCLE!_!BLOCK_LABEL!"
set "BLOCK_STATS=%TRAIN_STATS_DIR%\!BLOCK_NAME!.json"
set "BLOCK_LOG_DIR=%TRAIN_LOG_ROOT%\!BLOCK_NAME!"

if not exist "!BLOCK_LOG_DIR!" mkdir "!BLOCK_LOG_DIR!"

echo.
echo ------------------------------------------------------------
echo Training block !BLOCK_COUNTER! / 45
echo Cycle:       !CYCLE_NUMBER! / %CYCLES%
echo Opponent:    !BLOCK_OPPONENT!
echo Rounds:      !BLOCK_ROUNDS!
echo Block seed:  !BLOCK_SEED!
echo ------------------------------------------------------------
echo.

REM Always reactivate the latest preserved checkpoint before a new process.
copy /Y "%RESULT_DIR%\model.pt" "%ACTIVE_MODEL%" >nul
if errorlevel 1 (
    echo ERROR: Could not activate latest checkpoint before !BLOCK_NAME!.
    exit /b 1
)

python main.py play ^
    --no-gui ^
    --agents %AGENT% !BLOCK_OPPONENT! ^
    --train 1 ^
    --scenario classic ^
    --n-rounds !BLOCK_ROUNDS! ^
    --seed !BLOCK_SEED! ^
    --save-stats "!BLOCK_STATS!" ^
    --log-dir "!BLOCK_LOG_DIR!"

if errorlevel 1 (
    echo.
    echo ERROR: Training failed in !BLOCK_NAME!.
    exit /b 1
)

if exist "agent_code\%AGENT%\logs\%AGENT%.log" (
    copy /Y "agent_code\%AGENT%\logs\%AGENT%.log" "!BLOCK_LOG_DIR!\%AGENT%.log" >nul
)

if not exist "%ACTIVE_MODEL%" (
    echo ERROR: Training block completed but checkpoint was not found:
    echo   %ACTIVE_MODEL%
    exit /b 1
)

REM Save progress after every block. This also guarantees that the next block
REM resumes exactly from the model produced by this one.
copy /Y "%ACTIVE_MODEL%" "%RESULT_DIR%\model.pt" >nul
if errorlevel 1 (
    echo ERROR: Could not preserve checkpoint after !BLOCK_NAME!.
    exit /b 1
)

exit /b 0


REM ============================================================
REM SUBROUTINE: ONE EVALUATION OPPONENT
REM
REM Args:
REM   %1 = opponent name
REM   %2 = stats output path
REM   %3 = log directory
REM   %4 = short label
REM ============================================================

:EVAL_OPPONENT
set "EVAL_OPPONENT=%~1"
set "EVAL_STATS=%~2"
set "CURRENT_EVAL_LOG_DIR=%~3"
set "EVAL_LABEL=%~4"

if not exist "!CURRENT_EVAL_LOG_DIR!" mkdir "!CURRENT_EVAL_LOG_DIR!"

REM Reactivate the final model before each evaluation to guarantee that all
REM opponent evaluations use the identical checkpoint.
copy /Y "%RESULT_DIR%\model.pt" "%ACTIVE_MODEL%" >nul
if errorlevel 1 (
    echo ERROR: Could not activate final T4.2 checkpoint for !EVAL_LABEL! evaluation.
    exit /b 1
)

echo.
echo ------------------------------------------------------------
echo Evaluation vs !EVAL_OPPONENT!
echo Rounds: %EVAL_ROUNDS%
echo Seed:   %EVAL_SEED%
echo ------------------------------------------------------------
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% !EVAL_OPPONENT! ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %EVAL_ROUNDS% ^
    --seed %EVAL_SEED% ^
    --save-stats "!EVAL_STATS!" ^
    --log-dir "!CURRENT_EVAL_LOG_DIR!"

if errorlevel 1 (
    echo.
    echo ERROR: Evaluation failed against !EVAL_OPPONENT!.
    exit /b 1
)

if exist "agent_code\%AGENT%\logs\%AGENT%.log" (
    copy /Y "agent_code\%AGENT%\logs\%AGENT%.log" "!CURRENT_EVAL_LOG_DIR!\%AGENT%.log" >nul
)

exit /b 0
