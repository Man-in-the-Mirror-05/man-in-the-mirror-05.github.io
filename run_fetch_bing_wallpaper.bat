@echo off
REM 必应壁纸下载脚本的Windows批处理包装
REM 用于Windows任务计划程序

cd /d "%~dp0"
python fetch_bing_wallpaper.py

REM 如果Python脚本执行失败，记录错误
if errorlevel 1 (
    echo [%date% %time%] 脚本执行失败 >> fetch_error.log
    exit /b 1
)

exit /b 0

