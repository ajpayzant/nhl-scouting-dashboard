@echo off
REM Launch the NHL projection app. Double-click this file; a browser tab opens on its own.
REM PYTHONUTF8 is not optional: several hundred players have accented names and the
REM Windows console encoding will otherwise stop the run with a UnicodeEncodeError.
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
echo Starting the NHL projection app. Leave this window open; close it to stop the app.
python -m streamlit run app\streamlit_app.py
if errorlevel 1 (
  echo.
  echo The app did not start. If Streamlit is missing, run:  python -m pip install -r requirements.txt
  pause
)
endlocal
