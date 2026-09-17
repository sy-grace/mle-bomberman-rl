@echo off
setlocal

REM ============================================================
REM Task 4 tournament-style experiment runner
REM
REM Usage:
REM   run_task4.bat <agent> <feature_mode> <reward_mode> <experiment_seed>
REM   run_task4.bat <agent> <feature_mode> <reward_mode> <experiment_seed> [opponent_1] [opponent_2] [opponent_3] [gui]
REM
REM Defaults:
REM   agent=linear_q_agent
REM   feature_mode=f7
REM   reward_mode=shaped
REM   opponents=rule_based_agent rule_based_agent rule_based_agent
REM
REM Examples:
REM   run_task4.bat linear_q_agent f7 shaped 123
REM   run_task4.bat linear_q_agent f7 shaped 123 rule_based_agent coin_collector_agent rule_based_agent
REM   run_task4.bat linear_q_agent f7 shaped 123 rule_based_agent rule_based_agent rule_based_agent gui
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
set "OPPONENT_3=%~7"
set "GUI_EVAL=%~8"

set "TRAIN_ROUNDS=600"
set "EVAL_ROUNDS=100"
set "GUI_ROUNDS=10"
set "EVAL_SEED=999"

if "%AGENT%"=="" set "AGENT=linear_q_agent"
if "%FEATURE_MODE%"=="" set "FEATURE_MODE=f7"
if "%REWARD_MODE%"=="" set "REWARD_MODE=shaped"
if "%OPPONENT_1%"=="" set "OPPONENT_1=rule_based_agent"
if "%OPPONENT_2%"=="" set "OPPONENT_2=rule_based_agent"
if "%OPPONENT_3%"=="" set "OPPONENT_3=rule_based_agent"
if "%GUI_EVAL%"=="" set "GUI_EVAL=nogui"

if /I "%OPPONENT_1%"=="gui" (
    set "GUI_EVAL=gui"
    set "OPPONENT_1=rule_based_agent"
)
if /I "%OPPONENT_2%"=="gui" (
    set "GUI_EVAL=gui"
    set "OPPONENT_2=rule_based_agent"
)
if /I "%OPPONENT_3%"=="gui" (
    set "GUI_EVAL=gui"
    set "OPPONENT_3=rule_based_agent"
)

if /I not "%GUI_EVAL%"=="gui" if /I not "%GUI_EVAL%"=="nogui" (
    echo ERROR: GUI_EVAL must be gui or nogui.
    echo Usage: run_task4.bat ^<agent^> ^<f7^> ^<basic^|shaped^> ^<seed^> [opponent_1] [opponent_2] [opponent_3] [gui]
    exit /b 1
)

if "%EXPERIMENT_SEED%"=="" (
    echo ERROR: EXPERIMENT_SEED is missing.
    echo Usage: run_task4.bat ^<agent^> ^<f7^> ^<basic^|shaped^> ^<seed^> [opponent_1] [opponent_2] [opponent_3] [gui]
    exit /b 1
)

if /I not "%FEATURE_MODE%"=="f7" (
    echo ERROR: FEATURE_MODE must be f7 for Task 4 tournament experiments.
    exit /b 1
)

if /I not "%REWARD_MODE%"=="basic" if /I not "%REWARD_MODE%"=="shaped" (
    echo ERROR: REWARD_MODE must be basic or shaped.
    exit /b 1
)

if /I "%OPPONENT_1%"=="%AGENT%" (
    echo ERROR: OPPONENT_1 cannot be the same agent directory as AGENT in this runner.
    echo Use a copied variant directory if you want to self-play against an older checkpoint.
    exit /b 1
)
if /I "%OPPONENT_2%"=="%AGENT%" (
    echo ERROR: OPPONENT_2 cannot be the same agent directory as AGENT in this runner.
    echo Use a copied variant directory if you want to self-play against an older checkpoint.
    exit /b 1
)
if /I "%OPPONENT_3%"=="%AGENT%" (
    echo ERROR: OPPONENT_3 cannot be the same agent directory as AGENT in this runner.
    echo Use a copied variant directory if you want to self-play against an older checkpoint.
    exit /b 1
)

if /I "%GUI_EVAL%"=="gui" goto GUI_EVALUATION

set "RESULT_DIR=results\task4\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%\%OPPONENT_1%_%OPPONENT_2%_%OPPONENT_3%"

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
echo Task 4 tournament-style experiment
echo ============================================================
echo Agent:             %AGENT%
echo Feature mode:      %FEATURE_MODE%
echo Reward mode:       %REWARD_MODE%
echo Experiment seed:   %EXPERIMENT_SEED%
echo Opponent 1:        %OPPONENT_1%
echo Opponent 2:        %OPPONENT_2%
echo Opponent 3:        %OPPONENT_3%
echo Scenario:          classic
echo Training rounds:   %TRAIN_ROUNDS%
echo Evaluation rounds: %EVAL_ROUNDS%
echo Evaluation seed:   %EVAL_SEED%
echo Evaluation GUI:    %GUI_EVAL%
echo Result directory:  %RESULT_DIR%
echo ============================================================
echo.

set "MODEL_START_MODE=fresh"

echo Starting Task 4 training...
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENT_1% %OPPONENT_2% %OPPONENT_3% ^
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

set "MODEL_START_MODE=resume"

echo Starting Task 4 evaluation...
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENT_1% %OPPONENT_2% %OPPONENT_3% ^
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
echo Task 4 experiment finished successfully.
echo Results:
echo   %RESULT_DIR%
echo ============================================================
echo.

endlocal
exit /b 0

:GUI_EVALUATION
set "RESULT_DIR=results\task4\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%\%OPPONENT_1%_%OPPONENT_2%_%OPPONENT_3%"

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
echo Task 4 GUI evaluation only
echo ============================================================
echo Agent:             %AGENT%
echo Feature mode:      %FEATURE_MODE%
echo Reward mode:       %REWARD_MODE%
echo Training seed:     %EXPERIMENT_SEED%
echo Opponent 1:        %OPPONENT_1%
echo Opponent 2:        %OPPONENT_2%
echo Opponent 3:        %OPPONENT_3%
echo Scenario:          classic
echo GUI rounds:        %GUI_ROUNDS%
echo Evaluation seed:   %EVAL_SEED%
echo Checkpoint:        %RESULT_DIR%\model.pt
echo ============================================================
echo.

python main.py play ^
    --agents %AGENT% %OPPONENT_1% %OPPONENT_2% %OPPONENT_3% ^
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
