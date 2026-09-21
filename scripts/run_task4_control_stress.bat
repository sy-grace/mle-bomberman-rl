@echo off
setlocal EnableExtensions

REM ============================================================
REM Task 4.6 control stress evaluation
REM
REM Source model:
REM   T4.3 F7 control = T4.1 + 600 rounds against ONE rule_based_agent
REM
REM This adds no training. It evaluates that equal-training control
REM under the same multi-opponent conditions used for T4.6.
REM
REM Usage:
REM   scripts\run_task4_control_stress.bat <agent> <seed> <mixed|rule3> [run|eval|gui]
REM ============================================================

if "%~1"=="" goto :usage
if "%~2"=="" goto :usage
if "%~3"=="" goto :usage

set "AGENT=%~1"
set "BASE_SEED=%~2"
set "STRESS_MODE=%~3"
set "RUN_MODE=%~4"
if "%RUN_MODE%"=="" set "RUN_MODE=run"

set "FEATURE_MODE=f7"
set "REWARD_MODE=shaped"
set "MODEL_START_MODE=resume"
set "EXPERIMENT_SEED=%BASE_SEED%"
set "EVAL_ROUNDS=100"
set "GUI_ROUNDS=3"
set "EVAL_SEED=999"

if /I "%STRESS_MODE%"=="mixed" (
    set "VARIANT=control_stress_mixed"
    set "OPPONENTS=rule_based_agent coin_collector_agent peaceful_agent"
) else if /I "%STRESS_MODE%"=="rule3" (
    set "VARIANT=control_stress_rule3"
    set "OPPONENTS=rule_based_agent rule_based_agent rule_based_agent"
) else (
    echo ERROR: stress mode must be mixed or rule3.
    goto :usage
)

if /I not "%RUN_MODE%"=="run" if /I not "%RUN_MODE%"=="eval" if /I not "%RUN_MODE%"=="gui" (
    echo ERROR: mode must be run, eval, or gui.
    goto :usage
)

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul || (
    echo ERROR: Could not enter repository root.
    exit /b 1
)

set "SOURCE_DIR=results\task4\%AGENT%\f7_shaped_seed%BASE_SEED%_control"
set "SOURCE_MODEL=%SOURCE_DIR%\model.pt"
set "RESULT_DIR=results\task4\%AGENT%\f7_shaped_seed%BASE_SEED%_%VARIANT%"
set "ACTIVE_MODEL=agent_code\%AGENT%\my-saved-model.pt"

if not exist "%SOURCE_MODEL%" (
    echo ERROR: T4.3 equal-training control checkpoint not found:
    echo        %SOURCE_MODEL%
    popd >nul
    exit /b 1
)

if /I "%RUN_MODE%"=="run" (
    if exist "%RESULT_DIR%\eval.json" if exist "%RESULT_DIR%\model.pt" (
        echo Control stress %STRESS_MODE% seed %BASE_SEED% is already complete.
        popd >nul
        exit /b 0
    )
)

if /I "%RUN_MODE%"=="gui" goto :gui

if exist "%RESULT_DIR%" rmdir /S /Q "%RESULT_DIR%"
mkdir "%RESULT_DIR%" || goto :mkdir_error
mkdir "%RESULT_DIR%\eval_logs" || goto :mkdir_error

copy /Y "%SOURCE_MODEL%" "%RESULT_DIR%\model.pt" >nul || goto :copy_error
copy /Y "%SOURCE_MODEL%" "%ACTIVE_MODEL%" >nul || goto :copy_error
if exist "%SOURCE_DIR%\initial_model.pt" copy /Y "%SOURCE_DIR%\initial_model.pt" "%RESULT_DIR%\initial_model.pt" >nul
if exist "%SOURCE_DIR%\train.json" copy /Y "%SOURCE_DIR%\train.json" "%RESULT_DIR%\train.json" >nul

echo.
echo ============================================================
echo T4.6 equal-training control stress evaluation
echo ============================================================
echo Agent       : %AGENT%
echo Model seed  : %BASE_SEED%
echo Source      : %SOURCE_MODEL%
echo Condition   : %STRESS_MODE%
echo Opponents   : %OPPONENTS%
echo Eval rounds : %EVAL_ROUNDS%
echo Eval seed   : %EVAL_SEED%
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
    --log-dir "%RESULT_DIR%\eval_logs"

if errorlevel 1 (
    echo ERROR: Control stress evaluation failed.
    popd >nul
    exit /b 1
)

echo Control stress %STRESS_MODE% seed %BASE_SEED% completed successfully.
popd >nul
exit /b 0

:gui
copy /Y "%SOURCE_MODEL%" "%ACTIVE_MODEL%" >nul || goto :copy_error

python main.py play ^
    --agents %AGENT% %OPPONENTS% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %GUI_ROUNDS% ^
    --seed %EVAL_SEED%

set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%

:mkdir_error
echo ERROR: Could not create control stress result directory.
popd >nul
exit /b 1

:copy_error
echo ERROR: Failed to copy control stress checkpoint.
popd >nul
exit /b 1

:usage
echo Usage: %~nx0 ^<agent^> ^<seed^> ^<mixed^|rule3^> [run^|eval^|gui]
exit /b 1
