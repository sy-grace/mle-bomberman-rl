@echo off
setlocal

REM Task 1 SARSA lambda ablation
REM Compare SARSA(0), lambda=0.0, against the existing SARSA(lambda=0.8) results.
REM Fixed conditions: F1 features, basic reward, 600 training rounds,
REM 100 greedy evaluation rounds, train seeds 123/456/2026, eval seed 999.

set AGENT=sarsa_lambda_agent
set FEATURE_MODE=f1
set REWARD_MODE=basic
set SARSA_LAMBDA=0.0
set TRAIN_ROUNDS=600
set EVAL_ROUNDS=100
set EVAL_SEED=999

echo ============================================================
echo Task 1 SARSA lambda ablation
echo Agent:        %AGENT%
echo Feature mode: %FEATURE_MODE%
echo Reward mode:  %REWARD_MODE%
echo Lambda:       %SARSA_LAMBDA%
echo ============================================================
echo.

call :run_seed 123
if errorlevel 1 exit /b 1

call :run_seed 456
if errorlevel 1 exit /b 1

call :run_seed 2026
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo All 3 SARSA(0) ablation runs finished successfully.
echo Results: results\task1_lambda_ablation\%AGENT%
echo ============================================================

endlocal
exit /b 0


:run_seed
set TRAIN_SEED=%1
set RESULT_DIR=results\task1_lambda_ablation\%AGENT%\lambda0.0_seed%TRAIN_SEED%

echo.
echo ------------------------------------------------------------
echo Running SARSA(0): seed=%TRAIN_SEED%
echo ------------------------------------------------------------

REM Start with a clean result directory for this seed.
if exist "%RESULT_DIR%" rmdir /S /Q "%RESULT_DIR%"
mkdir "%RESULT_DIR%"
if errorlevel 1 (
    echo ERROR: Could not create result directory "%RESULT_DIR%".
    exit /b 1
)

REM Fresh training run.
set MODEL_START_MODE=fresh
set EXPERIMENT_SEED=%TRAIN_SEED%

python main.py play --no-gui --agents %AGENT% --train 1 --scenario coin-heaven --n-rounds %TRAIN_ROUNDS% --seed %TRAIN_SEED% --save-stats "%RESULT_DIR%\train.json"
if errorlevel 1 (
    echo ERROR: Training failed for seed %TRAIN_SEED%.
    exit /b 1
)

REM Preserve the trained checkpoint together with the raw statistics.
if not exist "agent_code\%AGENT%\my-saved-model.pt" (
    echo ERROR: Checkpoint was not created for seed %TRAIN_SEED%.
    exit /b 1
)

copy /Y "agent_code\%AGENT%\my-saved-model.pt" "%RESULT_DIR%\model.pt" >nul
if errorlevel 1 (
    echo ERROR: Could not copy checkpoint for seed %TRAIN_SEED%.
    exit /b 1
)

REM Greedy evaluation on the same fixed environment seed for all runs.
REM Evaluation loads the checkpoint produced immediately above.
set MODEL_START_MODE=resume
set EXPERIMENT_SEED=%EVAL_SEED%

python main.py play --no-gui --agents %AGENT% --train 0 --scenario coin-heaven --n-rounds %EVAL_ROUNDS% --seed %EVAL_SEED% --save-stats "%RESULT_DIR%\eval.json"
if errorlevel 1 (
    echo ERROR: Evaluation failed for seed %TRAIN_SEED%.
    exit /b 1
)

echo Finished seed %TRAIN_SEED%.
exit /b 0
