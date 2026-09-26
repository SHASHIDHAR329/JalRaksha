@echo off
cd /d "C:\JalRaksha\simulations\dflowfm\jalraksha_flume_scenario_04"
call "C:\Program Files\Deltares\Delft3D FM Suite 2026.01 HMWQ\plugins\DeltaShell.Dimr\kernels\x64\bin\run_dflowfm.bat" tidalflume.mdu
exit /b %errorlevel%
