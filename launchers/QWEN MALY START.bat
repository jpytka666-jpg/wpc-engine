@echo off
chcp 65001 >nul
title Qwen 4B - maly, lokalnie

REM Rozmowa z malym modelem na tym laptopie.
REM Qwen3-4B-Instruct spakowany do 4 bitow - 2 GB zamiast 15 GB duzego.
REM Odpowiada okolo 240 slow na minute, ponad piec razy szybciej niz 30B.
REM
REM PISZ PO ANGIELSKU. Po polsku maly model miesza jezyki - sprawdzone.
REM
REM Model siedzi po stronie Linuksa (WSL), wiec tam go uruchamiamy.
REM Windows Terminal, jesli jest - obsluguje polskie znaki i da sie kopiowac.
REM Sciezke skryptu mozna podmienic, gdyby przenioslo sie go gdzie indziej.

set "SKRYPT=/mnt/d/skrypty/qwen-maly.sh"
set "DYSTRYBUCJA=Ubuntu"

where wt >nul 2>&1
if %errorlevel%==0 (
    start "" wt -p "%DYSTRYBUCJA%" wsl -d %DYSTRYBUCJA% -- bash %SKRYPT%
    exit /b
)

echo.
echo   Maly Qwen 4B, lokalnie na tym laptopie.
echo   Pisz pytanie i Enter. Zeby wyjsc: koniec
echo   Najlepiej po angielsku.
echo.
wsl -d %DYSTRYBUCJA% -- bash %SKRYPT%
pause
