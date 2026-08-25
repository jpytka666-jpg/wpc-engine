@echo off
chcp 65001 >nul
title Qwen jako agent - z narzedziami AIONS

REM Qwen z rekami: siega po narzedzia AIONS i sam je wywoluje.
REM Rewolwer dobiera komore narzedzi pod zadanie (z 71 podaje kilka).
REM Brama zgody jest WLACZONA - nic sie nie wykona bez Twojego "y".
REM Zeby zakonczyc, model musi podac polecenie dowodzace, ze zrobil robote.

set "DYSTRYBUCJA=Ubuntu"
set "KATALOG=/home/aions/wpc-workspace"
set "MOST=/mnt/d/skrypty/aions_mcp.sh"

echo.
echo   Qwen jako agent, z narzedziami AIONS.
echo.
echo   Nic sie nie wykona bez Twojej zgody - pokaze, co chce zrobic,
echo   i poczeka az wciśniesz  y  i Enter. Cokolwiek innego to odmowa.
echo.

set /p ZADANIE=  Co ma zrobic?:

if "%ZADANIE%"=="" (
    echo   Nie podales zadania.
    pause
    exit /b
)

echo.
wsl -d %DYSTRYBUCJA% -- bash -lc "cd %KATALOG% && ./target/release/aions-agent --mcp-command bash --mcp-arg %MOST% --task '%ZADANIE%' --max-turns 4 --max-tokens 150 --ask"

echo.
pause
