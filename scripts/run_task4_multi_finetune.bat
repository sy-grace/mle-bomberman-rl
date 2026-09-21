@echo off
setlocal EnableExtensions

REM ============================================================
REM Task 4.6 - F7 SARSA multi-opponent combat fine-tuning
REM
REM Experimental question:
REM   Does +600 rounds of training against three simultaneous
REM   rule_based_agents improve hard multi-agent performance
REM   without sacrificing the retained T4.1 policy elsewhere?
REM
REM Parent checkpoint:
REM   results\task4\<agent>\f7_shaped_seed<seed>_finetuned\model.pt
REM
REM Training:
REM   SARSA + 3 x rule_based_agent
REM   F7, shaped, classic, 600 rounds
REM
REM Evaluation after training:
REM   1rule : 1 x rule_based_agent
REM   mixed : rule_based + coin_collector + peaceful
REM   rule3 : 3 x rule_based_agent
REM   100 rounds each, seed 999
REM
REM Usage:
REM   scripts\run_task4_multi_finetune.bat <agent> <seed> [run|eval|gui] [1rule|mixed|rule3]
REM
REM Examples:
REM   scripts\run_task4_multi_finetune.bat sarsa_lambda_agent 123 run
REM   scripts\run_task4_multi_finetune.bat sarsa_lambda_agent 123 eval
REM   scripts\run_task4_multi_finetune.bat sarsa_lambda_agent 123 gui rule3
REM ============================================================

if "%~1"=="" goto :usage
if "%~2"=="" goto :usage

set "AGENT=%~1"
set "BASE_SEED=%~2"
set "RUN_MODE=%~3"
set "GUI_CONDITION=%~4"

if "%RUN_MODE%"=="" set "RUN_MODE=run"
if "%GUI_CONDITION%"=="" set "GUI_CONDITION=rule3"

if /I not "%RUN_MODE%"=="run" if /I not "%RUN_MODE%"=="eval" if /I not "%RUN_MODE%"=="gui" (
    echo ERROR: mode must be run, eval, or gui.
    goto :usage
)

if /I not "%GUI_CONDITION%"=="1rule" if /I not "%GUI_CONDITION%"=="mixed" if /I not "%GUI_CONDITION%"=="rule3" (
    echo ERROR: GUI condition must be 1rule, mixed, or rule3.
    goto :usage
)

set "FEATURE_MODE=f7"
set "REWARD_MODE=shaped"
set "MODEL_START_MODE=resume"
set "EXPERIMENT_SEED=%BASE_SEED%"

set "TRAIN_ROUNDS=600"
set "EVAL_ROUNDS=100"
set "GUI_ROUNDS=3"
set "EVAL_SEED=999"

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul || (
    echo ERROR: Could not enter repository root.
    exit /b 1
)

set "SOURCE_DIR=results\task4\%AGENT%\f7_shaped_seed%BASE_SEED%_finetuned"
set "SOURCE_MODEL=%SOURCE_DIR%\model.pt"

REM Training artifacts live in a non-summary directory. Evaluation directories
REM below use the standard Task 4 naming convention and are summarized.
set "TRAIN_DIR=results\task4\%AGENT%\t46_multirule_seed%BASE_SEED%"
set "TRAIN_LOG_DIR=%TRAIN_DIR%\train_logs"
set "FINAL_MODEL=%TRAIN_DIR%\model.pt"
set "ACTIVE_MODEL=agent_code\%AGENT%\my-saved-model.pt"

set "EVAL_1RULE_DIR=results\task4\%AGENT%\f7_shaped_seed%BASE_SEED%_t46_1rule"
set "EVAL_MIXED_DIR=results\task4\%AGENT%\f7_shaped_seed%BASE_SEED%_t46_mixed"
set "EVAL_RULE3_DIR=results\task4\%AGENT%\f7_shaped_seed%BASE_SEED%_t46_rule3"

if not exist "%SOURCE_MODEL%" (
    echo ERROR: T4.1 source checkpoint not found:
    echo        %SOURCE_MODEL%
    popd >nul
    exit /b 1
)

if /I "%RUN_MODE%"=="run" goto :run
if /I "%RUN_MODE%"=="eval" goto :eval_only
if /I "%RUN_MODE%"=="gui" goto :gui

:run
REM A complete run has the trained model plus all three evaluations.
if exist "%FINAL_MODEL%" if exist "%EVAL_1RULE_DIR%\eval.json" if exist "%EVAL_MIXED_DIR%\eval.json" if exist "%EVAL_RULE3_DIR%\eval.json" (
    echo T4.6 seed %BASE_SEED% is already complete.
    echo Training model: %FINAL_MODEL%
    popd >nul
    exit /b 0
)

REM Partial runs are restarted from the exact T4.1 parent checkpoint.
if exist "%TRAIN_DIR%" rmdir /S /Q "%TRAIN_DIR%"
if exist "%EVAL_1RULE_DIR%" rmdir /S /Q "%EVAL_1RULE_DIR%"
if exist "%EVAL_MIXED_DIR%" rmdir /S /Q "%EVAL_MIXED_DIR%"
if exist "%EVAL_RULE3_DIR%" rmdir /S /Q "%EVAL_RULE3_DIR%"

mkdir "%TRAIN_DIR%" || goto :mkdir_error
mkdir "%TRAIN_LOG_DIR%" || goto :mkdir_error

copy /Y "%SOURCE_MODEL%" "%TRAIN_DIR%\initial_model.pt" >nul || goto :copy_error
copy /Y "%SOURCE_MODEL%" "%ACTIVE_MODEL%" >nul || goto :copy_error

echo.
echo ============================================================
echo T4.6 - multi-opponent combat fine-tuning
echo ============================================================
echo Agent             : %AGENT%
echo Feature           : f7
echo Reward            : shaped
echo Model seed        : %BASE_SEED%
echo Parent checkpoint : %SOURCE_MODEL%
echo Train opponents   : 3 x rule_based_agent
echo Train rounds      : %TRAIN_ROUNDS%
echo Train seed        : %BASE_SEED%
echo ============================================================
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% rule_based_agent rule_based_agent rule_based_agent ^
    --train 1 ^
    --scenario classic ^
    --n-rounds %TRAIN_ROUNDS% ^
    --seed %BASE_SEED% ^
    --save-stats "%TRAIN_DIR%\train.json" ^
    --log-dir "%TRAIN_LOG_DIR%"

if errorlevel 1 (
    echo ERROR: T4.6 training failed for seed %BASE_SEED%.
    popd >nul
    exit /b 1
)

if not exist "%ACTIVE_MODEL%" (
    echo ERROR: Training finished but active model was not found:
    echo        %ACTIVE_MODEL%
    popd >nul
    exit /b 1
)

copy /Y "%ACTIVE_MODEL%" "%FINAL_MODEL%" >nul || goto :copy_error

goto :evaluate_all

:eval_only
if not exist "%FINAL_MODEL%" (
    echo ERROR: T4.6 trained checkpoint not found:
    echo        %FINAL_MODEL%
    echo Run T4.6 with mode "run" first.
    popd >nul
    exit /b 1
)

goto :evaluate_all

:evaluate_all
call :evaluate_condition 1rule "%EVAL_1RULE_DIR%" "rule_based_agent"
if errorlevel 1 goto :eval_error

call :evaluate_condition mixed "%EVAL_MIXED_DIR%" "rule_based_agent coin_collector_agent peaceful_agent"
if errorlevel 1 goto :eval_error

call :evaluate_condition rule3 "%EVAL_RULE3_DIR%" "rule_based_agent rule_based_agent rule_based_agent"
if errorlevel 1 goto :eval_error

echo.
echo ============================================================
echo T4.6 seed %BASE_SEED% completed successfully.
echo Final trained model:
echo   %FINAL_MODEL%
echo Evaluations:
echo   %EVAL_1RULE_DIR%
echo   %EVAL_MIXED_DIR%
echo   %EVAL_RULE3_DIR%
echo ============================================================
echo.
popd >nul
exit /b 0

:evaluate_condition
set "COND_NAME=%~1"
set "COND_DIR=%~2"
set "COND_OPPONENTS=%~3"

if exist "%COND_DIR%" rmdir /S /Q "%COND_DIR%"
mkdir "%COND_DIR%" || exit /b 1
mkdir "%COND_DIR%\eval_logs" || exit /b 1

REM Store the exact T4.6 model used for this evaluation and duplicate train.json
REM so the summarizer can report the same training metrics on each evaluation row.
copy /Y "%FINAL_MODEL%" "%COND_DIR%\model.pt" >nul || exit /b 1
copy /Y "%TRAIN_DIR%\initial_model.pt" "%COND_DIR%\initial_model.pt" >nul || exit /b 1
if exist "%TRAIN_DIR%\train.json" copy /Y "%TRAIN_DIR%\train.json" "%COND_DIR%\train.json" >nul

REM Reactivate the untouched final trained model before every evaluation.
copy /Y "%FINAL_MODEL%" "%ACTIVE_MODEL%" >nul || exit /b 1

set "FEATURE_MODE=f7"
set "REWARD_MODE=shaped"
set "MODEL_START_MODE=resume"
set "EXPERIMENT_SEED=%BASE_SEED%"

echo.
echo ------------------------------------------------------------
echo T4.6 evaluation: %COND_NAME%
echo Opponents : %COND_OPPONENTS%
echo Rounds    : %EVAL_ROUNDS%
echo Eval seed : %EVAL_SEED%
echo Directory : %COND_DIR%
echo ------------------------------------------------------------
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% %COND_OPPONENTS% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %EVAL_ROUNDS% ^
    --seed %EVAL_SEED% ^
    --save-stats "%COND_DIR%\eval.json" ^
    --log-dir "%COND_DIR%\eval_logs"

exit /b %ERRORLEVEL%

:gui
if not exist "%FINAL_MODEL%" (
    echo ERROR: T4.6 trained checkpoint not found:
    echo        %FINAL_MODEL%
    echo Run T4.6 with mode "run" first.
    popd >nul
    exit /b 1
)

if /I "%GUI_CONDITION%"=="1rule" set "GUI_OPPONENTS=rule_based_agent"
if /I "%GUI_CONDITION%"=="mixed" set "GUI_OPPONENTS=rule_based_agent coin_collector_agent peaceful_agent"
if /I "%GUI_CONDITION%"=="rule3" set "GUI_OPPONENTS=rule_based_agent rule_based_agent rule_based_agent"

copy /Y "%FINAL_MODEL%" "%ACTIVE_MODEL%" >nul || goto :copy_error

set "FEATURE_MODE=f7"
set "REWARD_MODE=shaped"
set "MODEL_START_MODE=resume"
set "EXPERIMENT_SEED=%BASE_SEED%"

python main.py play ^
    --agents %AGENT% %GUI_OPPONENTS% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %GUI_ROUNDS% ^
    --seed %EVAL_SEED%

set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%

:eval_error
echo.
echo ERROR: T4.6 evaluation failed for seed %BASE_SEED%.
popd >nul
exit /b 1

:mkdir_error
echo ERROR: Could not create a T4.6 result directory.
popd >nul
exit /b 1

:copy_error
echo ERROR: Failed to copy a T4.6 checkpoint.
popd >nul
exit /b 1

:usage
echo Usage: %~nx0 ^<agent^> ^<seed^> [run^|eval^|gui] [1rule^|mixed^|rule3]
echo Example: %~nx0 sarsa_lambda_agent 123 run
exit /b 1
