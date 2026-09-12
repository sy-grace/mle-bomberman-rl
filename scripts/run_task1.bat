@echo off
setlocal

REM ============================================================
REM Task 1 experiment runner
REM
REM Usage:
REM   run_task1.bat <agent> <feature_mode> <reward_mode> <experiment_seed>
REM
REM Examples:
REM   run_task1.bat linear_q_agent f1 basic 101
REM   run_task1.bat sarsa_agent f1 shaped 202
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
    echo Usage: run_task1.bat ^<agent^> ^<f0^|f1^> ^<sparse^|basic^|shaped^> ^<seed^>
    exit /b 1
)

if "%FEATURE_MODE%"=="" (
    echo ERROR: FEATURE_MODE is missing.
    echo Usage: run_task1.bat ^<agent^> ^<f0^|f1^> ^<sparse^|basic^|shaped^> ^<seed^>
    exit /b 1
)

if "%REWARD_MODE%"=="" (
    echo ERROR: REWARD_MODE is missing.
    echo Usage: run_task1.bat ^<agent^> ^<f0^|f1^> ^<sparse^|basic^|shaped^> ^<seed^>
    exit /b 1
)

if "%EXPERIMENT_SEED%"=="" (
    echo ERROR: EXPERIMENT_SEED is missing.
    echo Usage: run_task1.bat ^<agent^> ^<f0^|f1^> ^<sparse^|basic^|shaped^> ^<seed^>
    exit /b 1
)

REM ------------------------------------------------------------
REM Validate feature mode
REM ------------------------------------------------------------

if /I not "%FEATURE_MODE%"=="f0" if /I not "%FEATURE_MODE%"=="f1" (
    echo ERROR: FEATURE_MODE must be f0 or f1.
    exit /b 1
)

REM ------------------------------------------------------------
REM Validate reward mode
REM ------------------------------------------------------------

if /I not "%REWARD_MODE%"=="sparse" if /I not "%REWARD_MODE%"=="basic" if /I not "%REWARD_MODE%"=="shaped" (
    echo ERROR: REWARD_MODE must be sparse, basic, or shaped.
    exit /b 1
)

REM ------------------------------------------------------------
REM Result directory
REM ------------------------------------------------------------

set "RESULT_DIR=results\task1\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%"

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
echo Task 1 experiment
echo ============================================================
echo Agent:             %AGENT%
echo Feature mode:      %FEATURE_MODE%
echo Reward mode:       %REWARD_MODE%
echo Experiment seed:   %EXPERIMENT_SEED%
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
set "REWARD_MODE=%REWARD_MODE%"

echo Starting training...
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% ^
    --train 1 ^
    --scenario coin-heaven ^
    --n-rounds %TRAIN_ROUNDS% ^
    --seed %EXPERIMENT_SEED% ^
    --save-stats "%RESULT_DIR%\train.json"

if errorlevel 1 (
    echo.
    echo ERROR: Training failed.
    exit /b 1
)

REM ------------------------------------------------------------
REM Preserve the trained checkpoint before another run can
REM overwrite my-saved-model.pt.
REM ------------------------------------------------------------

copy /Y "agent_code\%AGENT%\my-saved-model.pt" "%RESULT_DIR%\model.pt" >nul

if errorlevel 1 (
    echo.
    echo ERROR: Could not save checkpoint.
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
    --scenario coin-heaven ^
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
