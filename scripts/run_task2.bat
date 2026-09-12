@echo off
setlocal

REM ============================================================
REM Task 2 experiment runner
REM
REM Usage:
REM   run_task2.bat <agent> <feature_mode> <reward_mode> <experiment_seed>
REM
REM Examples:
REM   run_task2.bat linear_q_agent f2 shaped 123
REM   run_task2.bat linear_q_agent f3 shaped 123
REM   run_task2.bat sarsa_lambda_agent f2 basic 456
REM
REM Fixed settings:
REM   scenario=classic
REM   train=600 rounds
REM   eval=100 rounds
REM   eval seed=999
REM ============================================================

set "AGENT=%~1"
set "FEATURE_MODE=%~2"
set "REWARD_MODE=%~3"
set "EXPERIMENT_SEED=%~4"

set "TRAIN_ROUNDS=600"
set "EVAL_ROUNDS=100"
set "EVAL_SEED=999"

REM ------------------------------------------------------------
REM Validate required arguments
REM ------------------------------------------------------------

if "%AGENT%"=="" (
    echo ERROR: AGENT is missing.
    echo Usage: run_task2.bat ^<agent^> ^<f2^|f3^> ^<basic^|shaped^> ^<seed^>
    exit /b 1
)

if "%FEATURE_MODE%"=="" (
    echo ERROR: FEATURE_MODE is missing.
    echo Usage: run_task2.bat ^<agent^> ^<f2^|f3^> ^<basic^|shaped^> ^<seed^>
    exit /b 1
)

if "%REWARD_MODE%"=="" (
    echo ERROR: REWARD_MODE is missing.
    echo Usage: run_task2.bat ^<agent^> ^<f2^|f3^> ^<basic^|shaped^> ^<seed^>
    exit /b 1
)

if "%EXPERIMENT_SEED%"=="" (
    echo ERROR: EXPERIMENT_SEED is missing.
    echo Usage: run_task2.bat ^<agent^> ^<f2^|f3^> ^<basic^|shaped^> ^<seed^>
    exit /b 1
)

REM ------------------------------------------------------------
REM Validate feature / reward modes
REM ------------------------------------------------------------

if /I not "%FEATURE_MODE%"=="f2" if /I not "%FEATURE_MODE%"=="f3" (
    echo ERROR: FEATURE_MODE must be f2 or f3 for Task 2 experiments.
    exit /b 1
)

if /I not "%REWARD_MODE%"=="basic" if /I not "%REWARD_MODE%"=="shaped" (
    echo ERROR: REWARD_MODE must be basic or shaped for Task 2 experiments.
    exit /b 1
)

REM ------------------------------------------------------------
REM Result directory
REM ------------------------------------------------------------

set "RESULT_DIR=results\task2\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%"

REM Refuse to overwrite an existing completed/partial experiment.
if exist "%RESULT_DIR%\train.json" (
    echo ERROR: Training results already exist:
    echo   %RESULT_DIR%\train.json
    echo Delete or rename the existing result directory before rerunning.
    exit /b 1
)

if exist "%RESULT_DIR%\eval.json" (
    echo ERROR: Evaluation results already exist:
    echo   %RESULT_DIR%\eval.json
    echo Delete or rename the existing result directory before rerunning.
    exit /b 1
)

if exist "%RESULT_DIR%\model.pt" (
    echo ERROR: Saved model already exists:
    echo   %RESULT_DIR%\model.pt
    echo Delete or rename the existing result directory before rerunning.
    exit /b 1
)

if not exist "%RESULT_DIR%" (
    mkdir "%RESULT_DIR%"
)

if errorlevel 1 (
    echo ERROR: Could not create result directory:
    echo   %RESULT_DIR%
    exit /b 1
)

echo.
echo ============================================================
echo Task 2 experiment
echo ============================================================
echo Agent:             %AGENT%
echo Feature mode:      %FEATURE_MODE%
echo Reward mode:       %REWARD_MODE%
echo Experiment seed:   %EXPERIMENT_SEED%
echo Scenario:          classic
echo Training rounds:   %TRAIN_ROUNDS%
echo Evaluation rounds: %EVAL_ROUNDS%
echo Evaluation seed:   %EVAL_SEED%
echo Result directory:  %RESULT_DIR%
echo ============================================================
echo.

REM ============================================================
REM TRAINING
REM ============================================================

set "MODEL_START_MODE=fresh"

echo Starting training...
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% ^
    --train 1 ^
    --scenario classic ^
    --n-rounds %TRAIN_ROUNDS% ^
    --seed %EXPERIMENT_SEED% ^
    --save-stats "%RESULT_DIR%\train.json"

if errorlevel 1 (
    echo.
    echo ERROR: Training failed.
    exit /b 1
)

REM ------------------------------------------------------------
REM Preserve the trained checkpoint before another experiment can
REM overwrite agent_code\<agent>\my-saved-model.pt.
REM ------------------------------------------------------------

if not exist "agent_code\%AGENT%\my-saved-model.pt" (
    echo.
    echo ERROR: Training finished but checkpoint was not found:
    echo   agent_code\%AGENT%\my-saved-model.pt
    exit /b 1
)

copy /Y "agent_code\%AGENT%\my-saved-model.pt" "%RESULT_DIR%\model.pt" >nul

if errorlevel 1 (
    echo.
    echo ERROR: Could not preserve checkpoint.
    exit /b 1
)

echo.
echo Training finished successfully.
echo Checkpoint saved to:
echo   %RESULT_DIR%\model.pt
echo.

REM ============================================================
REM EVALUATION
REM ============================================================

set "MODEL_START_MODE=resume"

echo Starting evaluation...
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %EVAL_ROUNDS% ^
    --seed %EVAL_SEED% ^
    --save-stats "%RESULT_DIR%\eval.json"

if errorlevel 1 (
    echo.
    echo ERROR: Evaluation failed.
    exit /b 1
)

echo.
echo ============================================================
echo Experiment finished successfully.
echo Results:
echo   %RESULT_DIR%
echo ============================================================
echo.

endlocal
