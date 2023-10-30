@echo off
set "sourceFolder=."

if not exist "%sourceFolder%\177_Files\" mkdir "%sourceFolder%\train"

set count=0

for %%A in ("%sourceFolder%\*") do (
    set /a count+=1
    if !count! leq 177 (
        move "%%A" "%sourceFolder%\train"
    )
)

echo First 177 files moved to the '177_Files' subfolder.
pause
