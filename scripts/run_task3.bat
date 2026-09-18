@echo off
setlocal

REM ============================================================
REM Task 3 opponent-hunting experiment runner
REM
REM Usage:
REM   run_task3.bat <agent> <feature_mode> <reward_mode> <experiment_seed> <opponent>
REM   run_task3.bat <agent> <feature_mode> <reward_mode> <experiment_seed> <opponent> [gui]
REM
REM Opponents:
REM   peaceful_agent       Easy: moves randomly and never drops bombs.
REM   coin_collector_agent Hard: drops bombs to collect coins.
REM
REM Examples:
REM   run_task3.bat linear_q_agent f7 shaped 123 peaceful_agent
REM   run_task3.bat linear_q_agent f7 shaped 123 coin_collector_agent
REM   run_task3.bat linear_q_agent f7 shaped 123 peaceful_agent gui
REM ============================================================

set "AGENT=%~1"
set "FEATURE_MODE=%~2"
set "REWARD_MODE=%~3"
set "EXPERIMENT_SEED=%~4"
set "OPPONENT=%~5"
set "GUI_EVAL=%~6"

set "TRAIN_ROUNDS=600"
set "EVAL_ROUNDS=100"
set "GUI_ROUNDS=10"
set "EVAL_SEED=999"

if "%OPPONENT%"=="" set "OPPONENT=peaceful_agent"
if "%GUI_EVAL%"=="" set "GUI_EVAL=nogui"

if /I not "%GUI_EVAL%"=="gui" if /I not "%GUI_EVAL%"=="nogui" (
    echo ERROR: GUI_EVAL must be gui or nogui.
    exit /b 1
)

if "%AGENT%"=="" (
    echo ERROR: AGENT is missing.
    echo Usage: run_task3.bat ^<agent^> ^<f7^> ^<basic^|shaped^> ^<seed^> ^<opponent^> [gui]
    exit /b 1
)
if "%FEATURE_MODE%"=="" (
    echo ERROR: FEATURE_MODE is missing.
    exit /b 1
)
if "%REWARD_MODE%"=="" (
    echo ERROR: REWARD_MODE is missing.
    exit /b 1
)
if "%EXPERIMENT_SEED%"=="" (
    echo ERROR: EXPERIMENT_SEED is missing.
    exit /b 1
)

if /I not "%FEATURE_MODE%"=="f7" (
    echo ERROR: FEATURE_MODE must be f7 for Task 3.
    exit /b 1
)
if /I not "%REWARD_MODE%"=="basic" if /I not "%REWARD_MODE%"=="shaped" (
    echo ERROR: REWARD_MODE must be basic or shaped.
    exit /b 1
)
if /I not "%OPPONENT%"=="peaceful_agent" if /I not "%OPPONENT%"=="coin_collector_agent" (
    echo ERROR: OPPONENT must be peaceful_agent or coin_collector_agent.
    exit /b 1
)

set "RESULT_DIR=results\task3\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%\%OPPONENT%"

if /I "%GUI_EVAL%"=="gui" goto GUI_EVALUATION

for %%F in ("train.json" "eval.json" "model.pt") do (
    if exist "%RESULT_DIR%\%%~F" del /Q "%RESULT_DIR%\%%~F"
)
if not exist "%RESULT_DIR%" mkdir "%RESULT_DIR%"

set "MODEL_START_MODE=fresh"
echo Starting Task 3 training against %OPPONENT%...
call python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENT% ^
    --train 1 ^
    --scenario classic ^
    --n-rounds %TRAIN_ROUNDS% ^
    --seed %EXPERIMENT_SEED% ^
    --save-stats "%RESULT_DIR%\train.json"
if errorlevel 1 (
    echo ERROR: Training failed.
    exit /b 1
)

if not exist "agent_code\%AGENT%\my-saved-model.pt" (
    echo ERROR: Training checkpoint was not found.
    exit /b 1
)
copy /Y "agent_code\%AGENT%\my-saved-model.pt" "%RESULT_DIR%\model.pt" >nul
if errorlevel 1 (
    echo ERROR: Could not preserve the trained checkpoint.
    exit /b 1
)

set "MODEL_START_MODE=resume"
echo Starting Task 3 evaluation against %OPPONENT%...
call python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENT% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %EVAL_ROUNDS% ^
    --seed %EVAL_SEED% ^
    --save-stats "%RESULT_DIR%\eval.json"
if errorlevel 1 (
    echo ERROR: Evaluation failed.
    exit /b 1
)

echo Task 3 finished. Results saved in %RESULT_DIR%.
endlocal
exit /b 0

:GUI_EVALUATION
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
echo Starting Task 3 GUI evaluation against %OPPONENT%...
call python main.py play ^
    --agents %AGENT% %OPPONENT% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %GUI_ROUNDS% ^
    --seed %EVAL_SEED%
if errorlevel 1 (
    echo ERROR: GUI evaluation failed.
    exit /b 1
)

endlocal
