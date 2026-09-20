@echo off
setlocal

REM ============================================================
REM Task 4.3 control runner: continued F7 fine-tuning
REM
REM Purpose:
REM   Control for T4.3 F8 tactical features.
REM   Start from the same T4.1 F7 fine-tuned checkpoint and give
REM   the unchanged F7 agent the same additional 600 training
REM   rounds against rule_based_agent.
REM
REM Usage:
REM   run_task4_control.bat <agent> <reward_mode> <experiment_seed> [mode]
REM
REM Modes:
REM   run  = copy the matching T4.1 F7 checkpoint,
REM          continue F7 training for 600 rounds,
REM          then evaluate 100 rounds (default)
REM   eval = evaluate an existing control checkpoint only
REM   gui  = visualize an existing control checkpoint only
REM
REM Example:
REM   run_task4_control.bat sarsa_lambda_agent shaped 123 run
REM
REM Fixed settings (matched to T4.3 tactical runner):
REM   source feature = f7 (T4.1 finetuned checkpoint)
REM   target feature = f7 (unchanged control)
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

set "FEATURE_MODE=f7"
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
    echo Usage: run_task4_control.bat ^<agent^> ^<reward_mode^> ^<seed^> [run^|eval^|gui]
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
    echo Usage: run_task4_control.bat ^<agent^> ^<reward_mode^> ^<seed^> [run^|eval^|gui]
    exit /b 1
)

REM Keep the experiment identity fixed at the base seed.
set "EXPERIMENT_SEED=%BASE_SEED%"

set "SOURCE_DIR=results\task4\%AGENT%\f7_%REWARD_MODE%_seed%BASE_SEED%_finetuned"
set "SOURCE_CHECKPOINT=%SOURCE_DIR%\model.pt"
set "RESULT_DIR=results\task4\%AGENT%\f7_%REWARD_MODE%_seed%BASE_SEED%_control"
set "TRAIN_LOG_DIR=%RESULT_DIR%\train_logs"
set "EVAL_LOG_DIR=%RESULT_DIR%\eval_logs"
set "ACTIVE_CHECKPOINT=agent_code\%AGENT%\my-saved-model.pt"

if /I "%RUN_MODE%"=="eval" goto EVALUATION_ONLY
if /I "%RUN_MODE%"=="gui" goto GUI_EVALUATION

REM ============================================================
REM FULL CONTROL RUN: T4.1 F7 -> +600 F7 rounds -> eval
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
echo T4.3 CONTROL - continued F7 fine-tuning
echo ============================================================
echo Agent:             %AGENT%
echo Feature mode:      %FEATURE_MODE%
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
REM 1. Preserve and activate the exact T4.1 F7 checkpoint.
REM    No feature migration is performed in the control branch.
REM ------------------------------------------------------------

copy /Y "%SOURCE_CHECKPOINT%" "%RESULT_DIR%\initial_model.pt" >nul
if errorlevel 1 (
    echo ERROR: Could not preserve initial T4.1 F7 checkpoint.
    exit /b 1
)

copy /Y "%RESULT_DIR%\initial_model.pt" "%ACTIVE_CHECKPOINT%" >nul
if errorlevel 1 (
    echo ERROR: Could not activate initial F7 checkpoint.
    exit /b 1
)

set "MODEL_START_MODE=resume"

REM ------------------------------------------------------------
REM 2. Continue F7 training for the same 600 rounds used by F8.
REM ------------------------------------------------------------

echo.
echo Starting T4.3 control F7 fine-tuning...
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
    echo ERROR: T4.3 control F7 fine-tuning failed.
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
    echo ERROR: Could not preserve final T4.3 control checkpoint.
    exit /b 1
)

if exist "agent_code\%AGENT%\logs\%AGENT%.log" (
    copy /Y "agent_code\%AGENT%\logs\%AGENT%.log" "%TRAIN_LOG_DIR%\%AGENT%.log" >nul
)

REM ------------------------------------------------------------
REM 3. Evaluate final F7 control checkpoint with the same protocol.
REM ------------------------------------------------------------

set "MODEL_START_MODE=resume"

echo.
echo Starting T4.3 control evaluation...
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
    echo ERROR: T4.3 control evaluation failed.
    exit /b 1
)

if exist "agent_code\%AGENT%\logs\%AGENT%.log" (
    copy /Y "agent_code\%AGENT%\logs\%AGENT%.log" "%EVAL_LOG_DIR%\%AGENT%.log" >nul
)

echo.
echo ============================================================
echo T4.3 control finished successfully.
echo Initial T4.1 F7 checkpoint copy:
echo   %RESULT_DIR%\initial_model.pt
echo Final control checkpoint:
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
REM EVALUATE EXISTING CONTROL CHECKPOINT
REM ============================================================

:EVALUATION_ONLY

if not exist "%RESULT_DIR%\model.pt" (
    echo ERROR: T4.3 control checkpoint was not found:
    echo   %RESULT_DIR%\model.pt
    echo Run this experiment with mode "run" first.
    exit /b 1
)

if not exist "%EVAL_LOG_DIR%" mkdir "%EVAL_LOG_DIR%"

copy /Y "%RESULT_DIR%\model.pt" "%ACTIVE_CHECKPOINT%" >nul
if errorlevel 1 (
    echo ERROR: Could not activate T4.3 control checkpoint.
    exit /b 1
)

set "MODEL_START_MODE=resume"

echo.
echo ============================================================
echo T4.3 control evaluation only
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
    echo ERROR: T4.3 control evaluation-only run failed.
    exit /b 1
)

endlocal
exit /b 0


REM ============================================================
REM GUI EVALUATION OF EXISTING CONTROL CHECKPOINT
REM ============================================================

:GUI_EVALUATION

if not exist "%RESULT_DIR%\model.pt" (
    echo ERROR: T4.3 control checkpoint was not found:
    echo   %RESULT_DIR%\model.pt
    echo Run this experiment with mode "run" first.
    exit /b 1
)

copy /Y "%RESULT_DIR%\model.pt" "%ACTIVE_CHECKPOINT%" >nul
if errorlevel 1 (
    echo ERROR: Could not activate T4.3 control checkpoint.
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
    echo ERROR: T4.3 control GUI evaluation failed.
    exit /b 1
)

endlocal
exit /b 0
