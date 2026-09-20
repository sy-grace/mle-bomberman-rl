@echo off
setlocal EnableExtensions

REM ============================================================
REM Task 4.5 - multi-opponent stress test (evaluation only)
REM
REM Retained model:
REM   T4.1 F7 shaped fine-tuned checkpoint
REM
REM Conditions:
REM   mixed = rule_based_agent + coin_collector_agent + peaceful_agent
REM   rule3 = rule_based_agent + rule_based_agent + rule_based_agent
REM
REM No training is performed in T4.5.
REM
REM Usage:
REM   scripts\run_task4_stress.bat <agent> <seed> <mixed^|rule3> [run^|eval^|gui]
REM
REM Examples:
REM   scripts\run_task4_stress.bat sarsa_lambda_agent 123 mixed run
REM   scripts\run_task4_stress.bat sarsa_lambda_agent 123 rule3 run
REM ============================================================

if "%~1"=="" goto :usage
if "%~2"=="" goto :usage
if "%~3"=="" goto :usage

set "AGENT=%~1"
set "EXPERIMENT_SEED=%~2"
set "STRESS_MODE=%~3"
set "RUN_MODE=%~4"
if "%RUN_MODE%"=="" set "RUN_MODE=run"

set "FEATURE_MODE=f7"
set "REWARD_MODE=shaped"
set "MODEL_START_MODE=resume"
set "EVAL_ROUNDS=100"
set "GUI_ROUNDS=3"
set "EVAL_SEED=999"

if /I "%STRESS_MODE%"=="mixed" (
    set "VARIANT=stress_mixed"
    set "OPPONENTS=rule_based_agent coin_collector_agent peaceful_agent"
) else if /I "%STRESS_MODE%"=="rule3" (
    set "VARIANT=stress_rule3"
    set "OPPONENTS=rule_based_agent rule_based_agent rule_based_agent"
) else (
    echo ERROR: stress mode must be mixed or rule3.
    goto :usage
)

if /I not "%RUN_MODE%"=="run" if /I not "%RUN_MODE%"=="eval" if /I not "%RUN_MODE%"=="gui" (
    echo ERROR: mode must be run, eval, or gui.
    goto :usage
)

REM This script is expected to live in ^<repo^>\scripts.
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul || (
    echo ERROR: Could not enter repository root.
    exit /b 1
)

set "SOURCE_DIR=results\task4\%AGENT%\f7_shaped_seed%EXPERIMENT_SEED%_finetuned"
set "SOURCE_MODEL=%SOURCE_DIR%\model.pt"
set "RESULT_DIR=results\task4\%AGENT%\f7_shaped_seed%EXPERIMENT_SEED%_%VARIANT%"
set "ACTIVE_MODEL=agent_code\%AGENT%\my-saved-model.pt"
set "EVAL_LOG_DIR=%RESULT_DIR%\eval_logs"

if not exist "%SOURCE_MODEL%" (
    echo ERROR: T4.1 source checkpoint not found:
    echo        %SOURCE_MODEL%
    popd >nul
    exit /b 1
)

if /I "%RUN_MODE%"=="run" goto :run
if /I "%RUN_MODE%"=="eval" goto :eval
if /I "%RUN_MODE%"=="gui" goto :gui

:run
if exist "%RESULT_DIR%\eval.json" if exist "%RESULT_DIR%\model.pt" (
    echo T4.5 %STRESS_MODE% seed %EXPERIMENT_SEED% is already complete.
    echo Result: %RESULT_DIR%
    popd >nul
    exit /b 0
)

mkdir "%RESULT_DIR%" 2>nul
if exist "%EVAL_LOG_DIR%" rmdir /S /Q "%EVAL_LOG_DIR%"
mkdir "%EVAL_LOG_DIR%" 2>nul

REM Preserve the exact retained T4.1 checkpoint used for this stress test.
copy /Y "%SOURCE_MODEL%" "%RESULT_DIR%\initial_model.pt" >nul || goto :copy_error
copy /Y "%SOURCE_MODEL%" "%RESULT_DIR%\model.pt" >nul || goto :copy_error
copy /Y "%SOURCE_MODEL%" "%ACTIVE_MODEL%" >nul || goto :copy_error

goto :do_eval

:eval
mkdir "%RESULT_DIR%" 2>nul
if exist "%RESULT_DIR%\model.pt" (
    copy /Y "%RESULT_DIR%\model.pt" "%ACTIVE_MODEL%" >nul || goto :copy_error
) else (
    copy /Y "%SOURCE_MODEL%" "%RESULT_DIR%\initial_model.pt" >nul || goto :copy_error
    copy /Y "%SOURCE_MODEL%" "%RESULT_DIR%\model.pt" >nul || goto :copy_error
    copy /Y "%SOURCE_MODEL%" "%ACTIVE_MODEL%" >nul || goto :copy_error
)

if exist "%EVAL_LOG_DIR%" rmdir /S /Q "%EVAL_LOG_DIR%"
mkdir "%EVAL_LOG_DIR%" 2>nul
if exist "%RESULT_DIR%\eval.json" del /Q "%RESULT_DIR%\eval.json"

goto :do_eval

:do_eval
set "FEATURE_MODE=f7"
set "REWARD_MODE=shaped"
set "MODEL_START_MODE=resume"
set "EXPERIMENT_SEED=%EXPERIMENT_SEED%"

echo.
echo ============================================================
echo T4.5 multi-opponent stress evaluation
echo Agent       : %AGENT%
echo Model seed  : %EXPERIMENT_SEED%
echo Source      : %SOURCE_MODEL%
echo Feature     : %FEATURE_MODE%
echo Reward      : %REWARD_MODE%
echo Condition   : %STRESS_MODE%
echo Opponents   : %OPPONENTS%
echo Eval rounds : %EVAL_ROUNDS%
echo Eval seed   : %EVAL_SEED%
echo Result dir  : %RESULT_DIR%
echo ============================================================
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENTS% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %EVAL_ROUNDS% ^
    --seed %EVAL_SEED% ^
    --save-stats "%RESULT_DIR%\eval.json" ^
    --log-dir "%EVAL_LOG_DIR%"

if errorlevel 1 (
    echo ERROR: T4.5 evaluation failed.
    popd >nul
    exit /b 1
)

echo.
echo T4.5 %STRESS_MODE% seed %EXPERIMENT_SEED% completed successfully.
echo Result: %RESULT_DIR%
popd >nul
exit /b 0

:gui
if exist "%RESULT_DIR%\model.pt" (
    copy /Y "%RESULT_DIR%\model.pt" "%ACTIVE_MODEL%" >nul || goto :copy_error
) else (
    copy /Y "%SOURCE_MODEL%" "%ACTIVE_MODEL%" >nul || goto :copy_error
)

set "FEATURE_MODE=f7"
set "REWARD_MODE=shaped"
set "MODEL_START_MODE=resume"
set "EXPERIMENT_SEED=%EXPERIMENT_SEED%"

python main.py play ^
    --agents %AGENT% %OPPONENTS% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %GUI_ROUNDS% ^
    --seed %EVAL_SEED%

set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%

:copy_error
echo ERROR: Failed to copy checkpoint.
popd >nul
exit /b 1

:usage
echo Usage: %~nx0 ^<agent^> ^<seed^> ^<mixed^|rule3^> [run^|eval^|gui]
echo Example: %~nx0 sarsa_lambda_agent 123 mixed run
exit /b 1
