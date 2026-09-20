@echo off
setlocal

REM ============================================================
REM Task 4.3 tactical F8 experiment runner
REM
REM Usage:
REM   run_task4_tactical.bat <agent> <reward_mode> <experiment_seed> [mode]
REM
REM Modes:
REM   run  = migrate the matching T4.1 F7 checkpoint to F8,
REM          fine-tune 600 rounds vs rule_based_agent,
REM          then evaluate 100 rounds (default)
REM   eval = evaluate an existing T4.3 F8 checkpoint only
REM   gui  = visualize an existing T4.3 F8 checkpoint only
REM
REM Example:
REM   run_task4_tactical.bat sarsa_lambda_agent shaped 123 run
REM
REM Fixed settings:
REM   source feature = f7 (T4.1 finetuned checkpoint)
REM   target feature = f8
REM   scenario       = classic
REM   opponent       = rule_based_agent
REM   train          = 600 rounds
REM   eval           = 100 rounds
REM   eval seed      = 999
REM ============================================================

set "AGENT=%~1"
set "REWARD_MODE=%~2"
set "BASE_SEED=%~3"
set "RUN_MODE=%~4"

set "SOURCE_FEATURE=f7"
set "FEATURE_MODE=f8"
set "TRAIN_ROUNDS=600"
set "EVAL_ROUNDS=100"
set "GUI_ROUNDS=10"
set "EVAL_SEED=999"
set "OPPONENT=rule_based_agent"

if "%RUN_MODE%"=="" set "RUN_MODE=run"

REM ------------------------------------------------------------
REM Validate arguments
REM ------------------------------------------------------------

if "%AGENT%"=="" (
    echo ERROR: AGENT is missing.
    echo Usage: run_task4_tactical.bat ^<agent^> ^<reward_mode^> ^<seed^> [run^|eval^|gui]
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
    echo ERROR: REWARD_MODE must be basic or shaped.
    exit /b 1
)

if /I not "%RUN_MODE%"=="run" if /I not "%RUN_MODE%"=="eval" if /I not "%RUN_MODE%"=="gui" (
    echo ERROR: MODE must be run, eval, or gui.
    echo Usage: run_task4_tactical.bat ^<agent^> ^<reward_mode^> ^<seed^> [run^|eval^|gui]
    exit /b 1
)

REM Keep the experiment identity fixed at the base seed.
set "EXPERIMENT_SEED=%BASE_SEED%"

set "SOURCE_DIR=results\task4\%AGENT%\%SOURCE_FEATURE%_%REWARD_MODE%_seed%BASE_SEED%_finetuned"
set "SOURCE_CHECKPOINT=%SOURCE_DIR%\model.pt"
set "RESULT_DIR=results\task4\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%BASE_SEED%_tactical"
set "TRAIN_LOG_DIR=%RESULT_DIR%\train_logs"
set "EVAL_LOG_DIR=%RESULT_DIR%\eval_logs"
set "ACTIVE_CHECKPOINT=agent_code\%AGENT%\my-saved-model.pt"

if /I "%RUN_MODE%"=="eval" goto EVALUATION_ONLY
if /I "%RUN_MODE%"=="gui" goto GUI_EVALUATION

REM ============================================================
REM FULL T4.3 RUN: T4.1 F7 -> F8 migration -> training -> eval
REM ============================================================

if not exist "%SOURCE_CHECKPOINT%" (
    echo ERROR: T4.1 F7 fine-tuned checkpoint was not found:
    echo   %SOURCE_CHECKPOINT%
    echo Run T4.1 for this seed first.
    exit /b 1
)

if not exist "%RESULT_DIR%" (
    mkdir "%RESULT_DIR%"
    if errorlevel 1 (
        echo ERROR: Could not create result directory:
        echo   %RESULT_DIR%
        exit /b 1
    )
)

REM Remove outputs from an earlier incomplete/repeated run.
for %%F in ("train.json" "eval.json" "model.pt" "initial_model.pt") do (
    if exist "%RESULT_DIR%\%%~F" (
        echo Removing existing file:
        echo   %RESULT_DIR%\%%~F
        del /Q "%RESULT_DIR%\%%~F"
        if exist "%RESULT_DIR%\%%~F" (
            echo ERROR: Could not remove existing file:
            echo   %RESULT_DIR%\%%~F
            exit /b 1
        )
    )
)

if exist "%TRAIN_LOG_DIR%" rmdir /S /Q "%TRAIN_LOG_DIR%"
if exist "%EVAL_LOG_DIR%" rmdir /S /Q "%EVAL_LOG_DIR%"

mkdir "%TRAIN_LOG_DIR%"
if errorlevel 1 (
    echo ERROR: Could not create training log directory:
    echo   %TRAIN_LOG_DIR%
    exit /b 1
)

mkdir "%EVAL_LOG_DIR%"
if errorlevel 1 (
    echo ERROR: Could not create evaluation log directory:
    echo   %EVAL_LOG_DIR%
    exit /b 1
)

echo.
echo ============================================================
echo T4.3 - F8 opponent-aware tactical features
echo ============================================================
echo Agent:             %AGENT%
echo Source feature:    %SOURCE_FEATURE%
echo Target feature:    %FEATURE_MODE%
echo Reward mode:       %REWARD_MODE%
echo Experiment seed:   %BASE_SEED%
echo Source checkpoint: %SOURCE_CHECKPOINT%
echo Opponent:          %OPPONENT%
echo Scenario:          classic
echo Training rounds:   %TRAIN_ROUNDS%
echo Evaluation rounds: %EVAL_ROUNDS%
echo Evaluation seed:   %EVAL_SEED%
echo Result directory:  %RESULT_DIR%
echo ============================================================
echo.

REM ------------------------------------------------------------
REM 1. Expand the learned F7 model from 38 inputs to 42 inputs.
REM    Old weights are preserved; the four new F8 rows start at 0.
REM ------------------------------------------------------------

echo Migrating T4.1 F7 checkpoint to F8...
echo.

python -m scripts.migrate_f7_to_f8 "%SOURCE_CHECKPOINT%" "%RESULT_DIR%\initial_model.pt"

if errorlevel 1 (
    echo.
    echo ERROR: F7 to F8 checkpoint migration failed.
    exit /b 1
)

if not exist "%RESULT_DIR%\initial_model.pt" (
    echo ERROR: Migration finished but initial F8 checkpoint was not found:
    echo   %RESULT_DIR%\initial_model.pt
    exit /b 1
)

REM Activate the migrated F8 checkpoint for training.
copy /Y "%RESULT_DIR%\initial_model.pt" "%ACTIVE_CHECKPOINT%" >nul
if errorlevel 1 (
    echo ERROR: Could not activate migrated F8 checkpoint.
    exit /b 1
)

set "MODEL_START_MODE=resume"

REM ------------------------------------------------------------
REM 2. Fine-tune F8 for 600 rounds against rule_based_agent.
REM ------------------------------------------------------------

echo.
echo Starting T4.3 F8 fine-tuning...
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENT% ^
    --train 1 ^
    --scenario classic ^
    --n-rounds %TRAIN_ROUNDS% ^
    --seed %BASE_SEED% ^
    --save-stats "%RESULT_DIR%\train.json" ^
    --log-dir "%TRAIN_LOG_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: T4.3 F8 fine-tuning failed.
    exit /b 1
)

if not exist "%ACTIVE_CHECKPOINT%" (
    echo.
    echo ERROR: Training finished but active checkpoint was not found:
    echo   %ACTIVE_CHECKPOINT%
    exit /b 1
)

copy /Y "%ACTIVE_CHECKPOINT%" "%RESULT_DIR%\model.pt" >nul
if errorlevel 1 (
    echo.
    echo ERROR: Could not preserve final T4.3 checkpoint.
    exit /b 1
)

if exist "agent_code\%AGENT%\logs\%AGENT%.log" (
    copy /Y "agent_code\%AGENT%\logs\%AGENT%.log" "%TRAIN_LOG_DIR%\%AGENT%.log" >nul
)

REM ------------------------------------------------------------
REM 3. Evaluate final F8 checkpoint on the same Task 4 protocol.
REM ------------------------------------------------------------

set "MODEL_START_MODE=resume"

echo.
echo Starting T4.3 evaluation...
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENT% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %EVAL_ROUNDS% ^
    --seed %EVAL_SEED% ^
    --save-stats "%RESULT_DIR%\eval.json" ^
    --log-dir "%EVAL_LOG_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: T4.3 evaluation failed.
    exit /b 1
)

if exist "agent_code\%AGENT%\logs\%AGENT%.log" (
    copy /Y "agent_code\%AGENT%\logs\%AGENT%.log" "%EVAL_LOG_DIR%\%AGENT%.log" >nul
)

echo.
echo ============================================================
echo T4.3 finished successfully.
echo Migrated initial F8 checkpoint:
echo   %RESULT_DIR%\initial_model.pt
echo Final T4.3 checkpoint:
echo   %RESULT_DIR%\model.pt
echo Training stats:
echo   %RESULT_DIR%\train.json
echo Evaluation stats:
echo   %RESULT_DIR%\eval.json
echo Evaluation game log:
echo   %EVAL_LOG_DIR%\game.log
echo ============================================================
echo.

endlocal
exit /b 0


REM ============================================================
REM EVALUATE EXISTING T4.3 CHECKPOINT
REM ============================================================

:EVALUATION_ONLY

if not exist "%RESULT_DIR%\model.pt" (
    echo ERROR: T4.3 checkpoint was not found:
    echo   %RESULT_DIR%\model.pt
    echo Run this experiment with mode "run" first.
    exit /b 1
)

if not exist "%EVAL_LOG_DIR%" mkdir "%EVAL_LOG_DIR%"

copy /Y "%RESULT_DIR%\model.pt" "%ACTIVE_CHECKPOINT%" >nul
if errorlevel 1 (
    echo ERROR: Could not activate T4.3 checkpoint.
    exit /b 1
)

set "MODEL_START_MODE=resume"

echo.
echo ============================================================
echo T4.3 evaluation only
echo ============================================================
echo Agent:             %AGENT%
echo Feature mode:      %FEATURE_MODE%
echo Reward mode:       %REWARD_MODE%
echo Experiment seed:   %BASE_SEED%
echo Opponent:          %OPPONENT%
echo Evaluation rounds: %EVAL_ROUNDS%
echo Evaluation seed:   %EVAL_SEED%
echo Checkpoint:        %RESULT_DIR%\model.pt
echo ============================================================
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENT% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %EVAL_ROUNDS% ^
    --seed %EVAL_SEED% ^
    --save-stats "%RESULT_DIR%\eval.json" ^
    --log-dir "%EVAL_LOG_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: T4.3 evaluation-only run failed.
    exit /b 1
)

endlocal
exit /b 0


REM ============================================================
REM GUI EVALUATION OF EXISTING T4.3 CHECKPOINT
REM ============================================================

:GUI_EVALUATION

if not exist "%RESULT_DIR%\model.pt" (
    echo ERROR: T4.3 checkpoint was not found:
    echo   %RESULT_DIR%\model.pt
    echo Run this experiment with mode "run" first.
    exit /b 1
)

copy /Y "%RESULT_DIR%\model.pt" "%ACTIVE_CHECKPOINT%" >nul
if errorlevel 1 (
    echo ERROR: Could not activate T4.3 checkpoint.
    exit /b 1
)

set "MODEL_START_MODE=resume"

python main.py play ^
    --agents %AGENT% %OPPONENT% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %GUI_ROUNDS% ^
    --seed %EVAL_SEED%

if errorlevel 1 (
    echo.
    echo ERROR: T4.3 GUI evaluation failed.
    exit /b 1
)

endlocal
exit /b 0
