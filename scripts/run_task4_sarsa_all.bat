@echo off
setlocal

REM ============================================================
REM Run the three standard SARSA seeds for Task 4.
REM
REM Usage:
REM   run_task4_sarsa_all.bat [baseline|finetune|run]
REM
REM baseline (default): evaluate the matching Task 3 F7 checkpoints
REM                     unchanged against rule_based_agent.
REM finetune:           resume each matching Task 3 F7 checkpoint,
REM                     train 600 rounds against rule_based_agent,
REM                     then evaluate 100 rounds.
REM run:                train fresh F7 Task 4 agents against
REM                     rule_based_agent and then evaluate them.
REM
REM Add later Task 4 feature variants below once they exist.
REM ============================================================

set "AGENT=sarsa_lambda_agent"
set "FEATURE_MODE=f7"
set "REWARD_MODE=shaped"
set "RUN_MODE=%~1"

if "%RUN_MODE%"=="" set "RUN_MODE=baseline"

if /I not "%RUN_MODE%"=="baseline" if /I not "%RUN_MODE%"=="finetune" if /I not "%RUN_MODE%"=="run" (
    echo ERROR: MODE must be baseline, finetune, or run.
    echo Usage: run_task4_sarsa_all.bat [baseline^|finetune^|run]
    exit /b 1
)

echo ============================================================
echo Running Task 4 SARSA experiments - %FEATURE_MODE% %RUN_MODE%
echo Opponent: rule_based_agent
echo Completed experiments will be skipped.
echo ============================================================
echo.

call :RUN_EXPERIMENT 123
if errorlevel 1 exit /b 1

call :RUN_EXPERIMENT 456
if errorlevel 1 exit /b 1

call :RUN_EXPERIMENT 2026
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo All Task 4 SARSA %RUN_MODE% experiments finished successfully.
echo ============================================================

endlocal
exit /b 0


:RUN_EXPERIMENT
set "EXPERIMENT_SEED=%~1"

if /I "%RUN_MODE%"=="baseline" (
    set "RESULT_DIR=results\task4\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%_baseline"
) else if /I "%RUN_MODE%"=="finetune" (
    set "RESULT_DIR=results\task4\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%_finetuned"
) else (
    set "RESULT_DIR=results\task4\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%_trained"
)

echo.
echo ------------------------------------------------------------
echo Checking %FEATURE_MODE% seed %EXPERIMENT_SEED% - %RUN_MODE%
echo ------------------------------------------------------------

if /I "%RUN_MODE%"=="baseline" (
    if exist "%RESULT_DIR%\eval.json" if exist "%RESULT_DIR%\model.pt" (
        echo SKIP: Baseline already completed.
        echo   %RESULT_DIR%
        exit /b 0
    )
) else if /I "%RUN_MODE%"=="finetune" (
    if exist "%RESULT_DIR%\train.json" if exist "%RESULT_DIR%\eval.json" if exist "%RESULT_DIR%\model.pt" if exist "%RESULT_DIR%\initial_model.pt" (
        echo SKIP: Task 4 fine-tuning experiment already completed.
        echo   %RESULT_DIR%
        exit /b 0
    )
) else (
    if exist "%RESULT_DIR%\train.json" if exist "%RESULT_DIR%\eval.json" if exist "%RESULT_DIR%\model.pt" (
        echo SKIP: Fresh Task 4 training experiment already completed.
        echo   %RESULT_DIR%
        exit /b 0
    )
)

echo RUN: %FEATURE_MODE% seed %EXPERIMENT_SEED% - %RUN_MODE%
echo   %RESULT_DIR%
echo.

call scripts\run_task4.bat %AGENT% %FEATURE_MODE% %REWARD_MODE% %EXPERIMENT_SEED% %RUN_MODE%

if errorlevel 1 (
    echo.
    echo ERROR: %FEATURE_MODE% seed %EXPERIMENT_SEED% - %RUN_MODE% failed.
    exit /b 1
)

exit /b 0
