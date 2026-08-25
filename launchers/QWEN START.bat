@echo off
chcp 65001 >nul
title Qwen 30B - lokalnie

REM Rozmowa z wlasnym modelem na tym laptopie.
REM Model siedzi po stronie Linuksa (WSL), wiec tam go uruchamiamy.
REM Windows Terminal, jesli jest - obsluguje polskie znaki i da sie kopiowac.
REM Sciezke skryptu mozna podmienic, gdyby przenioslo sie go gdzie indziej.

set "SKRYPT=/mnt/d/skrypty/qwen.sh"
set "DYSTRYBUCJA=Ubuntu"

where wt >nul 2>&1
if %errorlevel%==0 (
    start "" wt -p "%DYSTRYBUCJA%" wsl -d %DYSTRYBUCJA% -- bash %SKRYPT%
    exit /b
)

echo.
echo   Qwen 30B, lokalnie na tym laptopie.
echo   Pisz pytanie i Enter. Zeby wyjsc: koniec
echo.
wsl -d %DYSTRYBUCJA% -- bash %SKRYPT%
pause
