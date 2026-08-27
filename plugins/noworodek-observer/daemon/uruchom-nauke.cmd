@echo off
REM ==========================================
REM AUTHOR: M. SZUL
REM AI MODEL: Claude Opus 5
REM TIMESTAMP: 2026-08-27 05:52:10
REM REASON FOR CREATION: A learning run must outlive the session that started it. Started
REM   from inside a Claude session it dies with that session, which turns a night of
REM   training into forty minutes of it. This is the one entry point: schedulable at
REM   logon, launchable detached, double-clickable, and the same each way.
REM MECHANICS: Runs the endless loop, which runs the conductor, which runs cycles - every
REM   layer below this still exits on its own, only this keeps going. If the loop dies on
REM   an unexpected fault it restarts after a minute, because a loop that stops on the
REM   first surprise was never endless. The STOP file halts everything including the
REM   restart. Never edit this file while it is running: Windows reads a command script
REM   line by line from disk, so replacing it mid-run resumes at the old byte offset in
REM   the new text - which is how a stray word became a command once already.
REM SYSTEM PART: Noworodek, training daemon.
REM ARCHITECTURE FUNCTION: The entry point the Windows scheduler calls at logon.
REM DEPENDENCIES/LINKS: wieczna-nauka.exe, karmiciel.exe, dyrygent.py, cykl.py, cbms.exe,
REM   train-cbms.exe, the shared code book.
REM TECH STACK: Windows command script - not a language choice but the scheduler's own
REM   calling convention; Rust cannot be what the scheduler launches without a wrapper,
REM   and this IS the wrapper. Everything it launches is compiled.
REM LOCAL WORKSPACE: daemon\uruchom-nauke.cmd
REM GIT COMMIT: PENDING
REM GITHUB METADATA: jpytka666-jpg/wpc-engine, branch noworodek-cbms-training
REM ==========================================
setlocal

set DAEMON=%USERPROFILE%\.claude\noworodek-observer\daemon
set CBMS=C:\temp\aions-cbms-2026-08-26\target\release\cbms.exe
set TRAINER=C:\temp\aions-cbms-train\target\release\train-cbms.exe
set LOG=%DAEMON%\wieczna-nauka.log

:petla
if exist "%DAEMON%\STOP" (
  echo [%date% %time%] plik STOP - nie startuje >> "%LOG%"
  goto :koniec
)

echo [%date% %time%] start petli >> "%LOG%"
"%DAEMON%\rust\target\release\wieczna-nauka.exe" --cbms "%CBMS%" --trainer "%TRAINER%" --cykli-na-runde 6 --minut-na-runde 90 --steps 800 >> "%LOG%" 2>&1
echo [%date% %time%] petla wyszla, kod %ERRORLEVEL% - restart za minute >> "%LOG%"

timeout /t 60 /nobreak >nul
goto :petla

:koniec
endlocal
