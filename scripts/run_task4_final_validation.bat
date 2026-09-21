@echo off
setlocal EnableExtensions

REM ============================================================
REM Task 4.7 - final SARSA model validation on unseen seeds
REM
REM Candidates:
REM   control123    = T4.3 F7 control, model seed 123
REM   finetuned456  = T4.1 F7 finetuned, model seed 456
REM
REM Fresh evaluation seeds:
REM   1001, 2002, 3003
REM
REM Conditions:
REM   1rule = 1 x rule_based_agent
REM   mixed = rule_based + coin_collector + peaceful
REM   rule3 = 3 x rule_based_agent
REM
REM No training is performed in T4.7.
REM
REM Usage:
REM   scripts\run_task4_final_validation.bat <agent> <candidate> <eval_seed> <condition> [run|gui]
REM
REM Examples:
REM   scripts\run_task4_final_validation.bat sarsa_lambda_agent control123 1001 rule3 run
REM   scripts\run_task4_final_validation.bat sarsa_lambda_agent finetuned456 2002 mixed run
REM ============================================================

if "%~1"=="" goto :usage
if "%~2"=="" goto :usage
if "%~3"=="" goto :usage
if "%~4"=="" goto :usage

set "AGENT=%~1"
set "CANDIDATE=%~2"
set "EVAL_SEED=%~3"
set "CONDITION=%~4"
set "RUN_MODE=%~5"
if "%RUN_MODE%"=="" set "RUN_MODE=run"

set "FEATURE_MODE=f7"
set "REWARD_MODE=shaped"
set "MODEL_START_MODE=resume"
set "EVAL_ROUNDS=100"
set "GUI_ROUNDS=3"

if /I "%CANDIDATE%"=="control123" (
    set "SOURCE_MODEL_SEED=123"
    set "SOURCE_DIR=results\task4\%AGENT%\f7_shaped_seed123_control"
    set "CANDIDATE_TAG=control123"
) else if /I "%CANDIDATE%"=="finetuned456" (
    set "SOURCE_MODEL_SEED=456"
    set "SOURCE_DIR=results\task4\%AGENT%\f7_shaped_seed456_finetuned"
    set "CANDIDATE_TAG=finetuned456"
) else (
    echo ERROR: candidate must be control123 or finetuned456.
    goto :usage
)

if /I "%CONDITION%"=="1rule" (
    set "CONDITION_TAG=1rule"
    set "OPPONENTS=rule_based_agent"
) else if /I "%CONDITION%"=="mixed" (
    set "CONDITION_TAG=mixed"
    set "OPPONENTS=rule_based_agent coin_collector_agent peaceful_agent"
) else if /I "%CONDITION%"=="rule3" (
    set "CONDITION_TAG=rule3"
    set "OPPONENTS=rule_based_agent rule_based_agent rule_based_agent"
) else (
    echo ERROR: condition must be 1rule, mixed, or rule3.
    goto :usage
)

if /I not "%RUN_MODE%"=="run" if /I not "%RUN_MODE%"=="gui" (
    echo ERROR: mode must be run or gui.
    goto :usage
)

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul || (
    echo ERROR: Could not enter repository root.
    exit /b 1
)

set "SOURCE_MODEL=%SOURCE_DIR%\model.pt"
set "VARIANT=t47_%CANDIDATE_TAG%_%CONDITION_TAG%"

REM IMPORTANT:
REM In T4.7 directory names, seedXXXX denotes the FRESH EVALUATION SEED,
REM not the model-training seed. The source model seed is encoded in VARIANT.
set "RESULT_DIR=results\task4\%AGENT%\f7_shaped_seed%EVAL_SEED%_%VARIANT%"
set "ACTIVE_MODEL=agent_code\%AGENT%\my-saved-model.pt"
set "EVAL_LOG_DIR=%RESULT_DIR%\eval_logs"

if not exist "%SOURCE_MODEL%" (
    echo ERROR: Candidate checkpoint not found:
    echo        %SOURCE_MODEL%
    popd >nul
    exit /b 1
)

if /I "%RUN_MODE%"=="gui" goto :gui

if exist "%RESULT_DIR%\eval.json" if exist "%RESULT_DIR%\model.pt" (
    echo T4.7 %CANDIDATE_TAG% / %CONDITION_TAG% / eval seed %EVAL_SEED% already complete.
    popd >nul
    exit /b 0
)

if exist "%RESULT_DIR%" rmdir /S /Q "%RESULT_DIR%"
mkdir "%RESULT_DIR%" || goto :mkdir_error
mkdir "%EVAL_LOG_DIR%" || goto :mkdir_error

REM Preserve the exact frozen checkpoint used in this evaluation.
copy /Y "%SOURCE_MODEL%" "%RESULT_DIR%\model.pt" >nul || goto :copy_error
copy /Y "%SOURCE_MODEL%" "%ACTIVE_MODEL%" >nul || goto :copy_error

REM The agent RNG is unused in non-training mode, but keep its identity tied
REM to the source model rather than the environment evaluation seed.
set "EXPERIMENT_SEED=%SOURCE_MODEL_SEED%"
set "FEATURE_MODE=f7"
set "REWARD_MODE=shaped"
set "MODEL_START_MODE=resume"

echo.
echo ============================================================
echo T4.7 - final validation
echo ============================================================
echo Candidate         : %CANDIDATE_TAG%
echo Source model seed : %SOURCE_MODEL_SEED%
echo Source checkpoint : %SOURCE_MODEL%
echo Feature           : f7
echo Reward            : shaped
echo Condition         : %CONDITION_TAG%
echo Opponents         : %OPPONENTS%
echo Eval rounds       : %EVAL_ROUNDS%
echo Fresh eval seed   : %EVAL_SEED%
echo Result directory  : %RESULT_DIR%
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
    echo ERROR: T4.7 evaluation failed.
    popd >nul
    exit /b 1
)

echo.
echo T4.7 completed successfully:
echo   Candidate : %CANDIDATE_TAG%
echo   Condition : %CONDITION_TAG%
echo   Eval seed : %EVAL_SEED%
echo.
popd >nul
exit /b 0

:gui
copy /Y "%SOURCE_MODEL%" "%ACTIVE_MODEL%" >nul || goto :copy_error

set "EXPERIMENT_SEED=%SOURCE_MODEL_SEED%"
set "FEATURE_MODE=f7"
set "REWARD_MODE=shaped"
set "MODEL_START_MODE=resume"

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
echo ERROR: Could not create T4.7 result directory.
popd >nul
exit /b 1

:copy_error
echo ERROR: Failed to copy candidate checkpoint.
popd >nul
exit /b 1

:usage
echo Usage: %~nx0 ^<agent^> ^<control123^|finetuned456^> ^<eval_seed^> ^<1rule^|mixed^|rule3^> [run^|gui]
echo Example: %~nx0 sarsa_lambda_agent control123 1001 rule3 run
exit /b 1
