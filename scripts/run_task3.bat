@echo off
setlocal

REM ============================================================
REM Task 3 experiment runner
REM
REM Usage:
REM   run_task3.bat <agent> <feature_mode> <reward_mode> <experiment_seed>
REM   run_task3.bat <agent> <feature_mode> <reward_mode> <experiment_seed> [gui]
REM
REM Add "gui" to visualize an existing trained checkpoint without
REM running training or overwriting experiment results.
REM
REM Examples:
REM   run_task3.bat sarsa_lambda_agent f5 shaped 123
REM   run_task3.bat sarsa_lambda_agent f6 shaped 456
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
set "GUI_EVAL=%~5"

set "TRAIN_ROUNDS=600"
set "EVAL_ROUNDS=100"
set "GUI_ROUNDS=10"
set "EVAL_SEED=999"

if "%GUI_EVAL%"=="" set "GUI_EVAL=nogui"

if /I not "%GUI_EVAL%"=="gui" if /I not "%GUI_EVAL%"=="nogui" (
    echo ERROR: GUI_EVAL must be gui or nogui.
    echo Usage: run_task3.bat ^<agent^> ^<f5^|f6^> ^<basic^|shaped^> ^<seed^> [gui]
    exit /b 1
)

REM ------------------------------------------------------------
REM Validate required arguments
REM ------------------------------------------------------------

if "%AGENT%"=="" (
    echo ERROR: AGENT is missing.
    echo Usage: run_task3.bat ^<agent^> ^<f5^|f6^> ^<basic^|shaped^> ^<seed^>
    exit /b 1
)

if "%FEATURE_MODE%"=="" (
    echo ERROR: FEATURE_MODE is missing.
    echo Usage: run_task3.bat ^<agent^> ^<f5^|f6^> ^<basic^|shaped^> ^<seed^>
    exit /b 1
)

if "%REWARD_MODE%"=="" (
    echo ERROR: REWARD_MODE is missing.
    echo Usage: run_task3.bat ^<agent^> ^<f5^|f6^> ^<basic^|shaped^> ^<seed^>
    exit /b 1
)

if "%EXPERIMENT_SEED%"=="" (
    echo ERROR: EXPERIMENT_SEED is missing.
    echo Usage: run_task3.bat ^<agent^> ^<f5^|f6^> ^<basic^|shaped^> ^<seed^>
    exit /b 1
)

REM ------------------------------------------------------------
REM Validate feature / reward modes
REM ------------------------------------------------------------

if /I not "%FEATURE_MODE%"=="f5" if /I not "%FEATURE_MODE%"=="f6" (
    echo ERROR: FEATURE_MODE must be f5 or f6 for Task 3 experiments.
    exit /b 1
)

if /I not "%REWARD_MODE%"=="basic" if /I not "%REWARD_MODE%"=="shaped" (
    echo ERROR: REWARD_MODE must be basic or shaped for Task 3 experiments.
    exit /b 1
)

if /I "%GUI_EVAL%"=="gui" goto GUI_EVALUATION

REM ------------------------------------------------------------
REM Result directory
REM ------------------------------------------------------------

set "RESULT_DIR=results\task3\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%"

REM Overwrite existing experiment artifacts.
for %%F in ("train.json" "eval.json" "model.pt") do (
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
echo Task 3 experiment
echo ============================================================
echo Agent:             %AGENT%
echo Feature mode:      %FEATURE_MODE%
echo Reward mode:       %REWARD_MODE%
echo Experiment seed:   %EXPERIMENT_SEED%
echo Scenario:          classic
echo Training rounds:   %TRAIN_ROUNDS%
echo Evaluation rounds: %EVAL_ROUNDS%
echo Evaluation seed:   %EVAL_SEED%
echo Evaluation GUI:   %GUI_EVAL%
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
    --agents %AGENT% peaceful_agent coin_collector_agent ^
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
exit /b 0

:GUI_EVALUATION
set "RESULT_DIR=results\task3\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%"

if not exist "%RESULT_DIR%\model.pt" (
    echo ERROR: Trained checkpoint was not found:
    echo   %RESULT_DIR%\model.pt
    echo Run the experiment without the gui option first.
    exit /b 1
)

copy /Y "%RESULT_DIR%\model.pt" "agent_code\%AGENT%\my-saved-model.pt" >nul

if errorlevel 1 (
    echo ERROR: Could not copy the trained checkpoint.
    exit /b 1
)

set "MODEL_START_MODE=resume"

echo.
echo ============================================================
echo GUI evaluation only
echo ============================================================
echo Agent:             %AGENT%
echo Feature mode:      %FEATURE_MODE%
echo Reward mode:       %REWARD_MODE%
echo Training seed:     %EXPERIMENT_SEED%
echo Scenario:          classic
echo GUI rounds:        %GUI_ROUNDS%
echo Evaluation seed:   %EVAL_SEED%
echo Checkpoint:        %RESULT_DIR%\model.pt
echo ============================================================
echo.

python main.py play ^
    --agents %AGENT% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %GUI_ROUNDS% ^
    --seed %EVAL_SEED%

if errorlevel 1 (
    echo.
    echo ERROR: GUI evaluation failed.
    exit /b 1
)

endlocal
