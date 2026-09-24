@echo off
setlocal
cd /d "%~dp0"

echo CUMTB-5G-Car Canny test tools
echo.
echo [1] CannyTester - image test
echo [2] CannyVideoTester - video test
echo [3] CannyFillScanTester - fill and scan test
echo [0] exit
echo.
set /p choice=Choose a tool: 

if "%choice%"=="1" start "CannyTester" "%~dp0dist\CannyTester.exe" & exit /b
if "%choice%"=="2" start "CannyVideoTester" "%~dp0dist\CannyVideoTester.exe" & exit /b
if "%choice%"=="3" start "CannyFillScanTester" "%~dp0dist\CannyFillScanTester.exe" & exit /b
if "%choice%"=="0" exit /b

echo Invalid choice.
pause
