@echo off
setlocal EnableExtensions

REM ============================================================
REM Task 4.4 - hunt-extra reward refinement
REM
REM Controlled experiment:
REM   start checkpoint : T4.1 F7 shaped fine-tuned model
REM   feature mode     : F7 (unchanged)
REM   opponent         : rule_based_agent (unchanged)
REM   training rounds  : 600 (unchanged)
REM   only change      : REWARD_MODE=hunt_extra
REM
REM Usage:
REM   scripts\run_task4_hunt_extra.bat <agent> <seed> [run^|eval^|gui]
REM
REM Example:
REM   scripts\run_task4_hunt_extra.bat sarsa_lambda_agent 123 run
REM ============================================================

if "%~1"=="" goto :usage
if "%~2"=="" goto :usage

set "AGENT=%~1"
set "EXPERIMENT_SEED=%~2"
set "RUN_MODE=%~3"
if "%RUN_MODE%"=="" set "RUN_MODE=run"

set "FEATURE_MODE=f7"
set "REWARD_MODE=hunt_extra"
set "MODEL_START_MODE=resume"
set "OPPONENT=rule_based_agent"
set "TRAIN_ROUNDS=600"
set "EVAL_ROUNDS=100"
set "EVAL_SEED=999"

REM This script is expected to live in <repo>\scripts.
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul || (
    echo ERROR: Could not enter repository root.
    exit /b 1
)

set "SOURCE_DIR=results\task4\%AGENT%\f7_shaped_seed%EXPERIMENT_SEED%_finetuned"
set "SOURCE_MODEL=%SOURCE_DIR%\model.pt"
set "RESULT_DIR=results\task4\%AGENT%\f7_hunt_extra_seed%EXPERIMENT_SEED%_t44"
set "ACTIVE_MODEL=agent_code\%AGENT%\my-saved-model.pt"
set "TRAIN_LOG_DIR=%RESULT_DIR%\train_logs"
set "EVAL_LOG_DIR=%RESULT_DIR%\eval_logs"

if /I "%RUN_MODE%"=="run" goto :run
if /I "%RUN_MODE%"=="eval" goto :eval
if /I "%RUN_MODE%"=="gui" goto :gui

echo ERROR: Unknown mode "%RUN_MODE%".
goto :usage_from_repo

:run
if not exist "%SOURCE_MODEL%" (
    echo ERROR: T4.1 source checkpoint not found:
    echo        %SOURCE_MODEL%
    popd >nul
    exit /b 1
)

if exist "%RESULT_DIR%\eval.json" if exist "%RESULT_DIR%\model.pt" (
    echo T4.4 seed %EXPERIMENT_SEED% is already complete.
    echo Result: %RESULT_DIR%
    popd >nul
    exit /b 0
)

if exist "%RESULT_DIR%\train.json" (
    echo ERROR: Partial T4.4 result already exists:
    echo        %RESULT_DIR%
    echo Delete or rename that directory before starting a clean controlled run.
    popd >nul
    exit /b 1
)

mkdir "%RESULT_DIR%" 2>nul
if exist "%TRAIN_LOG_DIR%" rmdir /S /Q "%TRAIN_LOG_DIR%"
if exist "%EVAL_LOG_DIR%" rmdir /S /Q "%EVAL_LOG_DIR%"
mkdir "%TRAIN_LOG_DIR%" 2>nul
mkdir "%EVAL_LOG_DIR%" 2>nul

REM Keep an immutable copy of the exact T4.1 starting checkpoint.
copy /Y "%SOURCE_MODEL%" "%RESULT_DIR%\initial_model.pt" >nul || goto :copy_error

REM Activate the same T4.1 model for continued F7 training.
copy /Y "%SOURCE_MODEL%" "%ACTIVE_MODEL%" >nul || goto :copy_error

set "FEATURE_MODE=%FEATURE_MODE%"
set "REWARD_MODE=%REWARD_MODE%"
set "MODEL_START_MODE=%MODEL_START_MODE%"
set "EXPERIMENT_SEED=%EXPERIMENT_SEED%"

echo.
echo ============================================================
echo T4.4 training
echo Agent      : %AGENT%
echo Seed       : %EXPERIMENT_SEED%
echo Source     : %SOURCE_MODEL%
echo Feature    : %FEATURE_MODE%
echo Reward     : %REWARD_MODE%
echo Opponent   : %OPPONENT%
echo Rounds     : %TRAIN_ROUNDS%
echo Result dir : %RESULT_DIR%
echo ============================================================
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENT% ^
    --train 1 ^
    --scenario classic ^
    --n-rounds %TRAIN_ROUNDS% ^
    --seed %EXPERIMENT_SEED% ^
    --save-stats "%RESULT_DIR%\train.json" ^
    --log-dir "%TRAIN_LOG_DIR%"

if errorlevel 1 (
    echo ERROR: T4.4 training failed.
    popd >nul
    exit /b 1
)

if not exist "%ACTIVE_MODEL%" (
    echo ERROR: Training finished but active checkpoint was not found:
    echo        %ACTIVE_MODEL%
    popd >nul
    exit /b 1
)

copy /Y "%ACTIVE_MODEL%" "%RESULT_DIR%\model.pt" >nul || goto :copy_error

goto :eval_after_train

:eval
if not exist "%RESULT_DIR%\model.pt" (
    echo ERROR: T4.4 trained checkpoint not found:
    echo        %RESULT_DIR%\model.pt
    popd >nul
    exit /b 1
)

if exist "%EVAL_LOG_DIR%" rmdir /S /Q "%EVAL_LOG_DIR%"
mkdir "%EVAL_LOG_DIR%" 2>nul
if exist "%RESULT_DIR%\eval.json" del /Q "%RESULT_DIR%\eval.json"
copy /Y "%RESULT_DIR%\model.pt" "%ACTIVE_MODEL%" >nul || goto :copy_error

goto :do_eval

:eval_after_train
REM Evaluate the checkpoint that was just trained.
if exist "%EVAL_LOG_DIR%" rmdir /S /Q "%EVAL_LOG_DIR%"
mkdir "%EVAL_LOG_DIR%" 2>nul
if exist "%RESULT_DIR%\eval.json" del /Q "%RESULT_DIR%\eval.json"
copy /Y "%RESULT_DIR%\model.pt" "%ACTIVE_MODEL%" >nul || goto :copy_error

:do_eval
set "FEATURE_MODE=f7"
set "REWARD_MODE=hunt_extra"
set "MODEL_START_MODE=resume"
set "EXPERIMENT_SEED=%EXPERIMENT_SEED%"

echo.
echo ============================================================
echo T4.4 evaluation
echo Agent      : %AGENT%
echo Seed       : %EXPERIMENT_SEED%
echo Eval seed  : %EVAL_SEED%
echo Rounds     : %EVAL_ROUNDS%
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
    echo ERROR: T4.4 evaluation failed.
    popd >nul
    exit /b 1
)

echo.
echo T4.4 seed %EXPERIMENT_SEED% completed successfully.
echo Result: %RESULT_DIR%
popd >nul
exit /b 0

:gui
if not exist "%RESULT_DIR%\model.pt" (
    echo ERROR: T4.4 trained checkpoint not found:
    echo        %RESULT_DIR%\model.pt
    popd >nul
    exit /b 1
)

copy /Y "%RESULT_DIR%\model.pt" "%ACTIVE_MODEL%" >nul || goto :copy_error

set "FEATURE_MODE=f7"
set "REWARD_MODE=hunt_extra"
set "MODEL_START_MODE=resume"
set "EXPERIMENT_SEED=%EXPERIMENT_SEED%"

python main.py play ^
    --agents %AGENT% %OPPONENT% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds 3 ^
    --seed %EVAL_SEED%

set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%

:copy_error
echo ERROR: Failed to copy checkpoint.
popd >nul
exit /b 1

:usage_from_repo
popd >nul

:usage
echo Usage: %~nx0 ^<agent^> ^<seed^> [run^|eval^|gui]
echo Example: %~nx0 sarsa_lambda_agent 123 run
exit /b 1
