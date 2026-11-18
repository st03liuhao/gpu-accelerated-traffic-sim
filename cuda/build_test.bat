@echo off
rem Builds the CUDA parity-test binary. Usage: cuda\build_test.bat [output.exe]
setlocal
set OUT=%~1
if "%OUT%"=="" set OUT=%~dp0test_lc_policy.exe
if not defined VCVARS set VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
call "%VCVARS%" >nul || exit /b 1
nvcc -O3 -std=c++17 -arch=native -I"%~dp0." "%~dp0test_lc_policy.cu" -o "%OUT%" || exit /b 1
