$ErrorActionPreference = "Stop"

Set-Location "C:\JalRaksha"

Write-Host ""
Write-Host "=============================================="
Write-Host "          JALRAKSHA PROTOTYPE"
Write-Host "=============================================="
Write-Host ""

conda activate jalraksha

Write-Host "Environment: jalraksha"
Write-Host "Starting Streamlit..."
Write-Host ""
Write-Host "Open: http://localhost:8501"
Write-Host ""

python -m streamlit run "C:\JalRaksha\src\app.py"
