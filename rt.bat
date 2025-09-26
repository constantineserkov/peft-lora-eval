@echo off
REM Activate uv environment (adjust path if needed)
call .venv\Scripts\activate.bat

REM Run training script
python -m scripts.run_train %*
