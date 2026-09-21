@echo off
setlocal

REM ============================================================
REM Task 3 experiment runner
REM
REM Usage:
REM   run_task3_three_agents.bat <agent> <feature_mode> <reward_mode> <experiment_seed> <opponent_1> <opponent_2>
REM   run_task3_three_agents.bat <agent> <feature_mode> <reward_mode> <experiment_seed> <opponent_1> <opponent_2> [gui]
REM
REM Opponents:
REM   peaceful_agent       Easy: moves randomly and never drops bombs.
REM   coin_collector_agent Hard: drops bombs to collect coins.
REM
REM Examples:
REM   run_task3_three_agents.bat linear_q_agent f7 shaped 123 coin_collector_agent peaceful_agent
REM   run_task3_three_agents.bat linear_q_agent f7 shaped 123 coin_collector_agent peaceful_agent gui
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
set "OPPONENT_1=%~5"
set "OPPONENT_2=%~6"
set "GUI_EVAL=%~7"

set "TRAIN_ROUNDS=600"
set "EVAL_ROUNDS=100"
set "GUI_ROUNDS=10"
set "EVAL_SEED=999"

if "%OPPONENT_1%"=="" set "OPPONENT_1=coin_collector_agent"
if "%OPPONENT_2%"=="" set "OPPONENT_2=peaceful_agent"
if "%GUI_EVAL%"=="" set "GUI_EVAL=nogui"

if /I not "%GUI_EVAL%"=="gui" if /I not "%GUI_EVAL%"=="nogui" (
    echo ERROR: GUI_EVAL must be gui or nogui.
    echo Usage: run_task3_three_agents.bat ^<agent^> ^<f7^> ^<basic^|shaped^> ^<seed^> ^<opponent_1^> ^<opponent_2^> [gui]
    exit /b 1
)

REM ------------------------------------------------------------
REM Validate required arguments
REM ------------------------------------------------------------

if "%AGENT%"=="" (
    echo ERROR: AGENT is missing.
    echo Usage: run_task3_three_agents.bat ^<agent^> ^<f7^> ^<basic^|shaped^> ^<seed^> ^<opponent_1^> ^<opponent_2^> [gui]
    exit /b 1
)

if "%FEATURE_MODE%"=="" (
    echo ERROR: FEATURE_MODE is missing.
    echo Usage: run_task3_three_agents.bat ^<agent^> ^<f7^> ^<basic^|shaped^> ^<seed^> ^<opponent_1^> ^<opponent_2^> [gui]
    exit /b 1
)

if "%REWARD_MODE%"=="" (
    echo ERROR: REWARD_MODE is missing.
    echo Usage: run_task3_three_agents.bat ^<agent^> ^<f7^> ^<basic^|shaped^> ^<seed^> ^<opponent_1^> ^<opponent_2^> [gui]
    exit /b 1
)

if "%EXPERIMENT_SEED%"=="" (
    echo ERROR: EXPERIMENT_SEED is missing.
    echo Usage: run_task3_three_agents.bat ^<agent^> ^<f7^> ^<basic^|shaped^> ^<seed^> ^<opponent_1^> ^<opponent_2^> [gui]
    exit /b 1
)

REM ------------------------------------------------------------
REM Validate feature / reward / opponent modes
REM ------------------------------------------------------------

if /I not "%FEATURE_MODE%"=="f7" (
    echo ERROR: FEATURE_MODE must be f5, f6, or f7 for Task 3 experiments.
    exit /b 1
)

if /I not "%REWARD_MODE%"=="basic" if /I not "%REWARD_MODE%"=="shaped" (
    echo ERROR: REWARD_MODE must be basic or shaped.
    exit /b 1
)

if /I not "%OPPONENT_1%"=="peaceful_agent" if /I not "%OPPONENT_1%"=="coin_collector_agent" (
    echo ERROR: OPPONENT_1 must be peaceful_agent or coin_collector_agent.
    exit /b 1
)

if /I not "%OPPONENT_2%"=="peaceful_agent" if /I not "%OPPONENT_2%"=="coin_collector_agent" (
    echo ERROR: OPPONENT_2 must be peaceful_agent or coin_collector_agent.
    exit /b 1
)

REM ------------------------------------------------------------
REM Build opponent arguments
REM ------------------------------------------------------------

set "OPPONENT_ARGS="

if /I "%OPPONENT%"=="both" (
    set "OPPONENT_ARGS=peaceful_agent coin_collector_agent"
)

if /I "%OPPONENT%"=="peaceful_agent" (
    set "OPPONENT_ARGS=peaceful_agent"
)

if /I "%OPPONENT%"=="coin_collector_agent" (
    set "OPPONENT_ARGS=coin_collector_agent"
)


REM ------------------------------------------------------------
REM Result directory
REM ------------------------------------------------------------

set "RESULT_DIR=results\task3\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%"

REM Keep the original master directory layout when using both
REM opponents. For single-opponent experiments, use a dedicated
REM subdirectory so checkpoints and statistics do not overwrite
REM each other.
if /I not "%OPPONENT%"=="both" (
    set "RESULT_DIR=results\task3\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%\%OPPONENT%"
)


REM ------------------------------------------------------------
REM Jump directly to evaluation / GUI if requested
REM ------------------------------------------------------------

if /I "%RUN_MODE%"=="eval" goto EVALUATION_ONLY
if /I "%RUN_MODE%"=="gui" goto GUI_EVALUATION

REM ------------------------------------------------------------
REM Result directory
REM ------------------------------------------------------------

set "RESULT_DIR=results\task3\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%\%OPPONENT_1%_%OPPONENT_2%"

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
echo Task 3 three-agent experiment
echo ============================================================
echo Agent:             %AGENT%
echo Feature mode:      %FEATURE_MODE%
echo Reward mode:       %REWARD_MODE%
echo Experiment seed:   %EXPERIMENT_SEED%
echo Opponent 1:        %OPPONENT_1%
echo Opponent 2:        %OPPONENT_2%
echo Scenario:          classic
echo Training rounds:   %TRAIN_ROUNDS%
echo Evaluation rounds: %EVAL_ROUNDS%
echo Evaluation seed:   %EVAL_SEED%
echo Evaluation GUI:    %GUI_EVAL%
echo Result directory:  %RESULT_DIR%
echo ============================================================
echo.

REM ============================================================
REM TRAINING
REM ============================================================

set "MODEL_START_MODE=fresh"

echo Starting Task 3 training against %OPPONENT_1% and %OPPONENT_2%...
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENT_1% %OPPONENT_2% ^
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

echo Starting Task 3 evaluation against %OPPONENT_1% and %OPPONENT_2%...
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENT_1% %OPPONENT_2% ^
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
echo Task 3 three-agent experiment finished successfully.
echo Results:
echo   %RESULT_DIR%
echo ============================================================
echo.

endlocal
exit /b 0



REM ============================================================
REM GUI EVALUATION ONLY
REM ============================================================

:GUI_EVALUATION
set "RESULT_DIR=results\task3\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%\%OPPONENT_1%_%OPPONENT_2%"

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
echo Task 3 three-agent GUI evaluation only
echo ============================================================
echo Agent:             %AGENT%
echo Feature mode:      %FEATURE_MODE%
echo Reward mode:       %REWARD_MODE%
echo Training seed:     %EXPERIMENT_SEED%
echo Opponent 1:        %OPPONENT_1%
echo Opponent 2:        %OPPONENT_2%
echo Scenario:          classic
echo GUI rounds:        %GUI_ROUNDS%
echo Evaluation seed:   %EVAL_SEED%
echo Checkpoint:        %RESULT_DIR%\model.pt
echo ============================================================
echo.

python main.py play ^
    --agents %AGENT% %OPPONENT_1% %OPPONENT_2% ^
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