@echo off
setlocal
cd /d "%~dp0"
set "WEB_GPU_PYTHON=%~dp0.venv-web-gpu\Scripts\python.exe"
if not exist "%WEB_GPU_PYTHON%" (
    echo Missing .venv-web-gpu. Follow GPU setup in WEB_GUIDE.md.
    exit /b 1
)
"%WEB_GPU_PYTHON%" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
if errorlevel 1 exit /b 1
"%WEB_GPU_PYTHON%" -m streamlit run app.py %*
