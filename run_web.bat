@echo off
setlocal
cd /d "%~dp0"
set "WEB_PYTHON=%~dp0.venv-web\Scripts\python.exe"
if not exist "%WEB_PYTHON%" (
    echo Missing .venv-web. Follow the Python setup in WEB_GUIDE.md.
    exit /b 1
)
"%WEB_PYTHON%" -c "import streamlit, ultralytics, cv2, imageio_ffmpeg, easyocr" >nul 2>&1
if errorlevel 1 (
    echo Missing web dependencies. Run:
    echo .\.venv-web\Scripts\python.exe -m pip install -r requirements-web.txt
    exit /b 1
)
"%WEB_PYTHON%" -m streamlit run app.py %*
