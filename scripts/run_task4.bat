@echo off
setlocal

REM ============================================================
REM Task 4 experiment runner
REM
REM Usage:
REM   run_task4.bat <agent> <feature_mode> <reward_mode> <experiment_seed> [mode]
REM
REM Modes:
REM   baseline = evaluate the matching Task 3 checkpoint against
REM              rule_based_agent without retraining (default)
REM   finetune = start from the matching Task 3 checkpoint, train
REM              600 rounds against rule_based_agent, then evaluate
REM              100 rounds
REM   run      = train 600 rounds from scratch against rule_based_agent,
REM              then evaluate 100 rounds (ablation)
REM   eval     = evaluate an existing fresh Task 4-trained checkpoint only
REM   gui      = visualize an existing fresh Task 4-trained checkpoint only
REM
REM Examples:
REM   run_task4.bat sarsa_lambda_agent f7 shaped 123 baseline
REM   run_task4.bat sarsa_lambda_agent f7 shaped 123 finetune
REM   run_task4.bat sarsa_lambda_agent f7 shaped 123 run
REM   run_task4.bat sarsa_lambda_agent f7 shaped 123 eval
REM   run_task4.bat sarsa_lambda_agent f7 shaped 123 gui
REM
REM Fixed settings:
REM   scenario=classic
REM   opponent=rule_based_agent
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
set "OPPONENT=rule_based_agent"

if "%RUN_MODE%"=="" set "RUN_MODE=baseline"

if /I not "%RUN_MODE%"=="baseline" if /I not "%RUN_MODE%"=="finetune" if /I not "%RUN_MODE%"=="run" if /I not "%RUN_MODE%"=="eval" if /I not "%RUN_MODE%"=="gui" (
    echo ERROR: MODE must be baseline, finetune, run, eval, or gui.
    echo Usage: run_task4.bat ^<agent^> ^<feature_mode^> ^<basic^|shaped^> ^<seed^> [baseline^|finetune^|run^|eval^|gui]
    exit /b 1
)

REM ------------------------------------------------------------
REM Validate required arguments
REM ------------------------------------------------------------

if "%AGENT%"=="" (
    echo ERROR: AGENT is missing.
    echo Usage: run_task4.bat ^<agent^> ^<feature_mode^> ^<basic^|shaped^> ^<seed^> [baseline^|finetune^|run^|eval^|gui]
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

if /I not "%REWARD_MODE%"=="basic" if /I not "%REWARD_MODE%"=="shaped" (
    echo ERROR: REWARD_MODE must be basic or shaped for Task 4 experiments.
    exit /b 1
)

REM ------------------------------------------------------------
REM Baseline: transfer the matching Task 3 checkpoint unchanged.
REM ------------------------------------------------------------

if /I "%RUN_MODE%"=="baseline" goto BASELINE_EVALUATION
if /I "%RUN_MODE%"=="finetune" goto FINETUNE_RUN

REM ------------------------------------------------------------
REM Fresh Task 4-trained experiment directories.
REM ------------------------------------------------------------

set "RESULT_DIR=results\task4\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%_trained"

if /I "%RUN_MODE%"=="eval" goto EVALUATION_ONLY
if /I "%RUN_MODE%"=="gui" goto GUI_EVALUATION

REM ============================================================
REM FULL RUN: FRESH TASK 4 TRAINING + EVALUATION
REM ============================================================

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

if not exist "%TRAIN_LOG_DIR%" (
    mkdir "%TRAIN_LOG_DIR%"
    if errorlevel 1 (
        echo ERROR: Could not create training log directory:
        echo   %TRAIN_LOG_DIR%
        exit /b 1
    )
)

if not exist "%EVAL_LOG_DIR%" (
    mkdir "%EVAL_LOG_DIR%"
    if errorlevel 1 (
        echo ERROR: Could not create evaluation log directory:
        echo   %EVAL_LOG_DIR%
        exit /b 1
    )
)

echo.
echo ============================================================
echo Task 4 experiment - fresh training
echo ============================================================
echo Agent:             %AGENT%
echo Feature mode:      %FEATURE_MODE%
echo Reward mode:       %REWARD_MODE%
echo Experiment seed:   %EXPERIMENT_SEED%
echo Scenario:          classic
echo Opponent:          %OPPONENT%
echo Training rounds:   %TRAIN_ROUNDS%
echo Evaluation rounds: %EVAL_ROUNDS%
echo Evaluation seed:   %EVAL_SEED%
echo Mode:              %RUN_MODE%
echo Result directory:  %RESULT_DIR%
echo ============================================================
echo.

set "MODEL_START_MODE=fresh"

echo Starting Task 4 training...
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENT% ^
    --train 1 ^
    --scenario classic ^
    --n-rounds %TRAIN_ROUNDS% ^
    --seed %EXPERIMENT_SEED% ^
    --save-stats "%RESULT_DIR%\train.json" ^
    --log-dir "%TRAIN_LOG_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: Task 4 training failed.
    exit /b 1
)

if exist "agent_code\%AGENT%\logs\%AGENT%.log" (
    copy /Y "agent_code\%AGENT%\logs\%AGENT%.log" "%TRAIN_LOG_DIR%\%AGENT%.log" >nul
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
    echo ERROR: Could not preserve Task 4 checkpoint.
    exit /b 1
)

set "MODEL_START_MODE=resume"

echo.
echo Starting Task 4 evaluation...
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENT% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %EVAL_ROUNDS% ^
    --seed %EVAL_SEED% ^
    --save-stats "%RESULT_DIR%\eval.json" ^
    --log-dir "%EVAL_LOG_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: Task 4 evaluation failed.
    exit /b 1
)

if exist "agent_code\%AGENT%\logs\%AGENT%.log" (
    copy /Y "agent_code\%AGENT%\logs\%AGENT%.log" "%EVAL_LOG_DIR%\%AGENT%.log" >nul
)

echo.
echo ============================================================
echo Task 4 experiment finished successfully.
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
REM TASK 3 -> TASK 4 FINE-TUNING
REM ============================================================

:FINETUNE_RUN

set "TASK3_RESULT_DIR=results\task3\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%"
set "RESULT_DIR=results\task4\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%_finetuned"
set "TRAIN_LOG_DIR=%RESULT_DIR%\train_logs"
set "EVAL_LOG_DIR=%RESULT_DIR%\eval_logs"

if not exist "%TASK3_RESULT_DIR%\model.pt" (
    echo ERROR: Matching Task 3 checkpoint was not found:
    echo   %TASK3_RESULT_DIR%\model.pt
    echo Run the corresponding Task 3 experiment first.
    exit /b 1
)

for %%F in ("train.json" "eval.json" "model.pt" "initial_model.pt") do (
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
        echo ERROR: Could not create Task 4 fine-tuning result directory:
        echo   %RESULT_DIR%
        exit /b 1
    )
)

if not exist "%TRAIN_LOG_DIR%" (
    mkdir "%TRAIN_LOG_DIR%"
    if errorlevel 1 (
        echo ERROR: Could not create fine-tuning log directory:
        echo   %TRAIN_LOG_DIR%
        exit /b 1
    )
)

if not exist "%EVAL_LOG_DIR%" (
    mkdir "%EVAL_LOG_DIR%"
    if errorlevel 1 (
        echo ERROR: Could not create fine-tuning evaluation log directory:
        echo   %EVAL_LOG_DIR%
        exit /b 1
    )
)

copy /Y "%TASK3_RESULT_DIR%\model.pt" "%RESULT_DIR%\initial_model.pt" >nul
if errorlevel 1 (
    echo ERROR: Could not preserve the Task 3 initialization checkpoint.
    exit /b 1
)

copy /Y "%TASK3_RESULT_DIR%\model.pt" "agent_code\%AGENT%\my-saved-model.pt" >nul
if errorlevel 1 (
    echo ERROR: Could not activate the Task 3 checkpoint for fine-tuning.
    exit /b 1
)

set "MODEL_START_MODE=resume"

echo.
echo ============================================================
echo Task 4 experiment - fine-tune Task 3 checkpoint
echo ============================================================
echo Agent:             %AGENT%
echo Feature mode:      %FEATURE_MODE%
echo Reward mode:       %REWARD_MODE%
echo Experiment seed:   %EXPERIMENT_SEED%
echo Source checkpoint: %TASK3_RESULT_DIR%\model.pt
echo Scenario:          classic
echo Opponent:          %OPPONENT%
echo Training rounds:   %TRAIN_ROUNDS%
echo Evaluation rounds: %EVAL_ROUNDS%
echo Evaluation seed:   %EVAL_SEED%
echo Mode:              %RUN_MODE%
echo Result directory:  %RESULT_DIR%
echo ============================================================
echo.

echo Starting Task 4 fine-tuning...
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENT% ^
    --train 1 ^
    --scenario classic ^
    --n-rounds %TRAIN_ROUNDS% ^
    --seed %EXPERIMENT_SEED% ^
    --save-stats "%RESULT_DIR%\train.json" ^
    --log-dir "%TRAIN_LOG_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: Task 4 fine-tuning failed.
    exit /b 1
)

if exist "agent_code\%AGENT%\logs\%AGENT%.log" (
    copy /Y "agent_code\%AGENT%\logs\%AGENT%.log" "%TRAIN_LOG_DIR%\%AGENT%.log" >nul
)

if not exist "agent_code\%AGENT%\my-saved-model.pt" (
    echo.
    echo ERROR: Fine-tuning finished but checkpoint was not found:
    echo   agent_code\%AGENT%\my-saved-model.pt
    exit /b 1
)

copy /Y "agent_code\%AGENT%\my-saved-model.pt" "%RESULT_DIR%\model.pt" >nul
if errorlevel 1 (
    echo.
    echo ERROR: Could not preserve fine-tuned Task 4 checkpoint.
    exit /b 1
)

set "MODEL_START_MODE=resume"

echo.
echo Starting Task 4 fine-tuned evaluation...
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENT% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %EVAL_ROUNDS% ^
    --seed %EVAL_SEED% ^
    --save-stats "%RESULT_DIR%\eval.json" ^
    --log-dir "%EVAL_LOG_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: Task 4 fine-tuned evaluation failed.
    exit /b 1
)

if exist "agent_code\%AGENT%\logs\%AGENT%.log" (
    copy /Y "agent_code\%AGENT%\logs\%AGENT%.log" "%EVAL_LOG_DIR%\%AGENT%.log" >nul
)

echo.
echo ============================================================
echo Task 4 fine-tuning finished successfully.
echo Initial Task 3 checkpoint:
echo   %RESULT_DIR%\initial_model.pt
echo Fine-tuned checkpoint:
echo   %RESULT_DIR%\model.pt
echo Evaluation:
echo   %RESULT_DIR%\eval.json
echo Evaluation game log:
echo   %EVAL_LOG_DIR%\game.log
echo ============================================================
echo.

endlocal
exit /b 0


REM ============================================================
REM TASK 3 -> TASK 4 TRANSFER BASELINE
REM ============================================================

:BASELINE_EVALUATION

set "TASK3_RESULT_DIR=results\task3\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%"
set "RESULT_DIR=results\task4\%AGENT%\%FEATURE_MODE%_%REWARD_MODE%_seed%EXPERIMENT_SEED%_baseline"
set "EVAL_LOG_DIR=%RESULT_DIR%\eval_logs"

if not exist "%TASK3_RESULT_DIR%\model.pt" (
    echo ERROR: Matching Task 3 checkpoint was not found:
    echo   %TASK3_RESULT_DIR%\model.pt
    echo Run the corresponding Task 3 experiment first.
    exit /b 1
)

if not exist "%RESULT_DIR%" (
    mkdir "%RESULT_DIR%"
    if errorlevel 1 (
        echo ERROR: Could not create Task 4 baseline result directory:
        echo   %RESULT_DIR%
        exit /b 1
    )
)

if not exist "%EVAL_LOG_DIR%" (
    mkdir "%EVAL_LOG_DIR%"
    if errorlevel 1 (
        echo ERROR: Could not create baseline evaluation log directory:
        echo   %EVAL_LOG_DIR%
        exit /b 1
    )
)

copy /Y "%TASK3_RESULT_DIR%\model.pt" "%RESULT_DIR%\model.pt" >nul
if errorlevel 1 (
    echo ERROR: Could not preserve the Task 3 checkpoint in the Task 4 baseline directory.
    exit /b 1
)

copy /Y "%TASK3_RESULT_DIR%\model.pt" "agent_code\%AGENT%\my-saved-model.pt" >nul
if errorlevel 1 (
    echo ERROR: Could not activate the Task 3 checkpoint for Task 4 baseline evaluation.
    exit /b 1
)

set "MODEL_START_MODE=resume"

echo.
echo ============================================================
echo Task 4 baseline - unchanged Task 3 checkpoint
echo ============================================================
echo Agent:             %AGENT%
echo Feature mode:      %FEATURE_MODE%
echo Reward mode:       %REWARD_MODE%
echo Training seed:     %EXPERIMENT_SEED%
echo Source checkpoint: %TASK3_RESULT_DIR%\model.pt
echo Scenario:          classic
echo Opponent:          %OPPONENT%
echo Evaluation rounds: %EVAL_ROUNDS%
echo Evaluation seed:   %EVAL_SEED%
echo Result directory:  %RESULT_DIR%
echo ============================================================
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENT% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %EVAL_ROUNDS% ^
    --seed %EVAL_SEED% ^
    --save-stats "%RESULT_DIR%\eval.json" ^
    --log-dir "%EVAL_LOG_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: Task 4 baseline evaluation failed.
    exit /b 1
)

if exist "agent_code\%AGENT%\logs\%AGENT%.log" (
    copy /Y "agent_code\%AGENT%\logs\%AGENT%.log" "%EVAL_LOG_DIR%\%AGENT%.log" >nul
)

echo.
echo ============================================================
echo Task 4 baseline finished successfully.
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
REM EVALUATE EXISTING TASK 4-TRAINED CHECKPOINT
REM ============================================================

:EVALUATION_ONLY

if not exist "%RESULT_DIR%\model.pt" (
    echo ERROR: Task 4-trained checkpoint was not found:
    echo   %RESULT_DIR%\model.pt
    echo Run Task 4 with mode "run" first.
    exit /b 1
)

set "EVAL_LOG_DIR=%RESULT_DIR%\eval_logs"
if not exist "%EVAL_LOG_DIR%" (
    mkdir "%EVAL_LOG_DIR%"
    if errorlevel 1 (
        echo ERROR: Could not create evaluation log directory:
        echo   %EVAL_LOG_DIR%
        exit /b 1
    )
)

copy /Y "%RESULT_DIR%\model.pt" "agent_code\%AGENT%\my-saved-model.pt" >nul
if errorlevel 1 (
    echo ERROR: Could not activate the Task 4-trained checkpoint.
    exit /b 1
)

set "MODEL_START_MODE=resume"

echo.
echo ============================================================
echo Task 4 evaluation only
echo ============================================================
echo Agent:             %AGENT%
echo Feature mode:      %FEATURE_MODE%
echo Reward mode:       %REWARD_MODE%
echo Training seed:     %EXPERIMENT_SEED%
echo Opponent:          %OPPONENT%
echo Evaluation rounds: %EVAL_ROUNDS%
echo Evaluation seed:   %EVAL_SEED%
echo Checkpoint:        %RESULT_DIR%\model.pt
echo ============================================================
echo.

python main.py play ^
    --no-gui ^
    --agents %AGENT% %OPPONENT% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %EVAL_ROUNDS% ^
    --seed %EVAL_SEED% ^
    --save-stats "%RESULT_DIR%\eval.json" ^
    --log-dir "%EVAL_LOG_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: Task 4 evaluation-only run failed.
    exit /b 1
)

if exist "agent_code\%AGENT%\logs\%AGENT%.log" (
    copy /Y "agent_code\%AGENT%\logs\%AGENT%.log" "%EVAL_LOG_DIR%\%AGENT%.log" >nul
)

endlocal
exit /b 0


REM ============================================================
REM GUI EVALUATION OF EXISTING TASK 4-TRAINED CHECKPOINT
REM ============================================================

:GUI_EVALUATION

if not exist "%RESULT_DIR%\model.pt" (
    echo ERROR: Task 4-trained checkpoint was not found:
    echo   %RESULT_DIR%\model.pt
    echo Run Task 4 with mode "run" first.
    exit /b 1
)

copy /Y "%RESULT_DIR%\model.pt" "agent_code\%AGENT%\my-saved-model.pt" >nul
if errorlevel 1 (
    echo ERROR: Could not activate the Task 4-trained checkpoint.
    exit /b 1
)

set "MODEL_START_MODE=resume"

echo.
echo ============================================================
echo Task 4 GUI evaluation
echo ============================================================
echo Agent:           %AGENT%
echo Feature mode:    %FEATURE_MODE%
echo Reward mode:     %REWARD_MODE%
echo Training seed:   %EXPERIMENT_SEED%
echo Opponent:        %OPPONENT%
echo GUI rounds:      %GUI_ROUNDS%
echo Evaluation seed: %EVAL_SEED%
echo Checkpoint:      %RESULT_DIR%\model.pt
echo ============================================================
echo.

python main.py play ^
    --agents %AGENT% %OPPONENT% ^
    --train 0 ^
    --scenario classic ^
    --n-rounds %GUI_ROUNDS% ^
    --seed %EVAL_SEED%

if errorlevel 1 (
    echo.
    echo ERROR: Task 4 GUI evaluation failed.
    exit /b 1
)

endlocal
exit /b 0
