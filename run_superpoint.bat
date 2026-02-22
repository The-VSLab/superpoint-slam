@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ================================================================================
echo  🚀 SuperPoint SLAM 자동 실행
echo ================================================================================
echo.

cd /d "%~dp0"

set "PYTHON_exe=C:\Users\dlcjs\AppData\Local\Programs\Python\Python311\python.exe"
set "INPUT=.\assets\test2.mp4"
set "WEIGHTS=.\weights\superpoint_cocoms.pth"
set "MODE=compare"
set "MIN_KPTS=600"
set "MAX_KPTS=600"
set "KPT_RADIUS=1"
set "BOTTOM_RATIO=0.35"

REM 명령어 
set "CMD=%PYTHON_exe% .\scripts\superpoint_app.py"
set "CMD=!CMD! --mode %MODE%"
set "CMD=!CMD! --input %INPUT%"
set "CMD=!CMD! --weights %WEIGHTS%"
set "CMD=!CMD! --aggressive_shadow_filter"
set "CMD=!CMD! --min_kpts %MIN_KPTS%"
set "CMD=!CMD! --max_kpts %MAX_KPTS%"
set "CMD=!CMD! --kpt_display_radius %KPT_RADIUS%"
set "CMD=!CMD! --bottom_region_ratio %BOTTOM_RATIO%"

echo 📋 설정:
echo   MODE: %MODE%
echo   INPUT: %INPUT%
echo   WEIGHTS: %WEIGHTS%
echo   MIN_KEYPOINTS: %MIN_KPTS%
echo   MAX_KEYPOINTS: %MAX_KPTS%
echo   RADIUS: %KPT_RADIUS%px
echo   BOTTOM_RATIO: %BOTTOM_RATIO%
echo.
echo ⏳ 실행 중... (프로그램을 닫으려면 Ctrl+C)
echo.

%CMD%

echo.
echo ================================================================================
echo  ✅ 완료!
echo ================================================================================
echo  📁 결과 폴더: result\superpoint_XX (자동 번호 매김)
echo ================================================================================
pause
