@echo off
setlocal

REM ============================================================
REM Task 3 experiment runner
REM
REM Usage:
REM   run_task3.bat <agent> <feature_mode> <reward_mode> <experiment_seed>
REM   run_task3.bat <agent> <feature_mode> <reward_mode> <experiment_seed> [mode]
REM
REM Modes:
REM   run   = train 600 rounds + evaluate 100 rounds (default)
REM   eval  = evaluate an existing checkpoint only; do not retrain
REM   gui   = visualize an existing checkpoint only
REM
REM "nogui" is accepted as a backwards-compatible alias for "run".
REM
REM Examples:
REM   run_task3.bat sarsa_lambda_agent f5 shaped 123
REM   run_task3.bat sarsa_lambda_agent f6 shaped 456
REM   run_task3.bat sarsa_lambda_agent f7 shaped 2026
REM   run_task3.bat sarsa_lambda_agent f5 shaped 123 eval
REM   run_task3.bat sarsa_lambda_agent f6 shaped 456 gui
REM
REM Fixed settings:
REM   scenario=classic
REM   opponents=peaceful_agent + coin_collector_agent
REM   train=600 rounds
REM   eval=100 rounds
REM   eval seed=999
REM ============================================================

set "AGENT=%~1"
set "FEATURE_MODE=%~2"
set "REWARD_MODE=%~3"
set "EXPERIMENT_SEED=%~4"
set "RUN_MODE=%~5"

set "TRAIN_ROUNDS=600"
set "EVAL_ROUNDS=100"
set "GUI_ROUNDS=10"
set "EVAL_SEED=999"

if "%RUN_MODE%"=="" set "RUN_MODE=run"
if /I "%RUN_MODE%"=="nogui" set "RUN_MODE=run"

if /I not "%RUN_MODE%"=="run" if /I not "%RUN_MODE%"=="eval" if /I not "%RUN_MODE%"=="gui" (
    echo ERROR: MODE must be run, eval, or gui.
    echo Usage: run_task3.bat ^<agent^> ^<f5^|f6^|f7^> ^<basic^|shaped^> ^<seed^> [run^|eval^|gui]
    exit /b 1
)

REM ------------------------------------------------------------
REM Validate required arguments
REM ------------------------------------------------------------

if "%AGENT%"=="" (
    echo ERROR: AGENT is missing.
    echo Usage: run_task3.bat ^<agent^> ^<f5^|f6^|f7^> ^<basic^|shaped^> ^<seed^> [run^|eval^|gui]
    exit /b 1
)

if "%FEATURE_MODE%"=="" (
    echo ERROR: FEATURE_MODE is missing.
    echo Usage: run_task3.bat ^<agent^> ^<f5^|f6^|f7^> ^<basic^|shaped^> ^<seed^> [run^|eval^|gui]
    exit /b 1
)

if "%REWARD_MODE%"=="" (
    echo ERROR: REWARD_MODE is missing.
    echo Usage: run_task3.bat ^<agent^> ^<f5^|f6^|f7^> ^<basic^|shaped^> ^<seed^> [run^|eval^|gui]
    exit /b 1
)

if "%EXPERIMENT_SEED%"=="" (
    echo ERROR: EXPERIMENT_SEED is missing.
    echo Usage: run_task3.bat ^<agent^> ^<f5^|f6^|f7^> ^<basic^|shaped^> ^<seed^> [run^|eval^|gui]
    exit /b 1
)

REM ------------------------------------------------------------
REM Validate feature / reward modes
REM ------------------------------------------------------------

if /I not "%FEATURE_MODE%"=="f5" if /I not "%FEATURE_MODE%"=="f6" if /I not "%FEATURE_MODE%"=="f7" (
    echo ERROR: FEATURE_MODE must be f5, f6, or f7 for Task 3 experiments.
    exit /b 1
)

if /I not "%REWARD_MODE%"=="basic" if /I not "%REWARD_MODE%"=="shaped" (
    echo ERROR: REWARD_MODE must be basic or shaped for Task 3 experiments.
    exit /b 1
)

REM ------------------------------------------------------------
REM Result directory
REM ------------------------------------------------------------

set "RESULT_DIR=results\task3\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%"

if /I "%RUN_MODE%"=="eval" goto EVALUATION_ONLY
if /I "%RUN_MODE%"=="gui" goto GUI_EVALUATION

REM ============================================================
REM FULL RUN: TRAINING + EVALUATION
REM ============================================================

REM Overwrite only the main experiment artifacts.
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
    if errorlevel 1 (
        echo ERROR: Could not create result directory:
        echo   %RESULT_DIR%
        exit /b 1
    )
)

set "TRAIN_LOG_DIR=%RESULT_DIR%\train_logs"
set "EVAL_LOG_DIR=%RESULT_DIR%\eval_logs"

if not exist "%TRAIN_LOG_DIR%" mkdir "%TRAIN_LOG_DIR%"
if errorlevel 1 (
    echo ERROR: Could not create training log directory:
    echo   %TRAIN_LOG_DIR%
    exit /b 1
)

if not exist "%EVAL_LOG_DIR%" mkdir "%EVAL_LOG_DIR%"
if errorlevel 1 (
    echo ERROR: Could not create evaluation log directory:
    echo   %EVAL_LOG_DIR%
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
echo Opponents:         peaceful_agent + coin_collector_agent
echo Training rounds:   %TRAIN_ROUNDS%
echo Evaluation rounds: %EVAL_ROUNDS%
echo Evaluation seed:   %EVAL_SEED%
echo Mode:              %RUN_MODE%
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
    --save-stats "%RESULT_DIR%\train.json" ^
    --log-dir "%TRAIN_LOG_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: Training failed.
    exit /b 1
)

REM Preserve our agent-code log because the framework writes it to
REM agent_code\<agent>\logs rather than to --log-dir.
if exist "agent_code\%AGENT%\logs\%AGENT%.log" (
    copy /Y "agent_code\%AGENT%\logs\%AGENT%.log" "%TRAIN_LOG_DIR%\%AGENT%.log" >nul
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
echo Training logs saved to:
echo   %TRAIN_LOG_DIR%
echo.

REM ============================================================
REM EVALUATION
REM ============================================================

set "MODEL_START_MODE=resume"

echo Starting evaluation...
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% peaceful_agent coin_collector_agent ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %EVAL_ROUNDS% ^
    --seed %EVAL_SEED% ^
    --save-stats "%RESULT_DIR%\eval.json" ^
    --log-dir "%EVAL_LOG_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: Evaluation failed.
    exit /b 1
)

if exist "agent_code\%AGENT%\logs\%AGENT%.log" (
    copy /Y "agent_code\%AGENT%\logs\%AGENT%.log" "%EVAL_LOG_DIR%\%AGENT%.log" >nul
)

echo.
echo ============================================================
echo Experiment finished successfully.
echo Results:
echo   %RESULT_DIR%
echo Evaluation game log:
echo   %EVAL_LOG_DIR%\game.log
echo Evaluation agent log:
echo   %EVAL_LOG_DIR%\%AGENT%.log
echo ============================================================
echo.

endlocal
exit /b 0


REM ============================================================
REM EVALUATION ONLY
REM ============================================================

:EVALUATION_ONLY

if not exist "%RESULT_DIR%\model.pt" (
    echo ERROR: Trained checkpoint was not found:
    echo   %RESULT_DIR%\model.pt
    echo Run the full experiment first.
    exit /b 1
)

if not exist "%RESULT_DIR%" (
    echo ERROR: Result directory was not found:
    echo   %RESULT_DIR%
    exit /b 1
)

set "DEATH_LOG_DIR=%RESULT_DIR%\eval_death_logs"

if not exist "%DEATH_LOG_DIR%" (
    mkdir "%DEATH_LOG_DIR%"
    if errorlevel 1 (
        echo ERROR: Could not create death-analysis log directory:
        echo   %DEATH_LOG_DIR%
        exit /b 1
    )
)

copy /Y "%RESULT_DIR%\model.pt" "agent_code\%AGENT%\my-saved-model.pt" >nul

if errorlevel 1 (
    echo ERROR: Could not copy the trained checkpoint.
    exit /b 1
)

set "MODEL_START_MODE=resume"

echo.
echo ============================================================
echo Evaluation only - death analysis
echo ============================================================
echo Agent:             %AGENT%
echo Feature mode:      %FEATURE_MODE%
echo Reward mode:       %REWARD_MODE%
echo Training seed:     %EXPERIMENT_SEED%
echo Scenario:          classic
echo Opponents:         peaceful_agent + coin_collector_agent
echo Evaluation rounds: %EVAL_ROUNDS%
echo Evaluation seed:   %EVAL_SEED%
echo Checkpoint:        %RESULT_DIR%\model.pt
echo ============================================================
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% peaceful_agent coin_collector_agent ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %EVAL_ROUNDS% ^
    --seed %EVAL_SEED% ^
    --save-stats "%RESULT_DIR%\eval_death_analysis.json" ^
    --log-dir "%DEATH_LOG_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: Evaluation-only run failed.
    exit /b 1
)

if exist "agent_code\%AGENT%\logs\%AGENT%.log" (
    copy /Y "agent_code\%AGENT%\logs\%AGENT%.log" "%DEATH_LOG_DIR%\%AGENT%.log" >nul
)

echo.
echo ============================================================
echo Death-analysis evaluation finished successfully.
echo Stats:
echo   %RESULT_DIR%\eval_death_analysis.json
echo Game log:
echo   %DEATH_LOG_DIR%\game.log
echo Agent log:
echo   %DEATH_LOG_DIR%\%AGENT%.log
echo ============================================================
echo.

endlocal
exit /b 0


REM ============================================================
REM GUI EVALUATION ONLY
REM ============================================================

:GUI_EVALUATION

if not exist "%RESULT_DIR%\model.pt" (
    echo ERROR: Trained checkpoint was not found:
    echo   %RESULT_DIR%\model.pt
    echo Run the full experiment first.
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
echo Opponents:         peaceful_agent + coin_collector_agent
echo GUI rounds:        %GUI_ROUNDS%
echo Evaluation seed:   %EVAL_SEED%
echo Checkpoint:        %RESULT_DIR%\model.pt
echo ============================================================
echo.

python main.py play ^
    --agents %AGENT% peaceful_agent coin_collector_agent ^
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
exit /b 0
