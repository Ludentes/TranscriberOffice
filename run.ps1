# run.ps1 - Start the Meeting Transcriber server
# Run with: powershell -ExecutionPolicy Bypass -File run.ps1

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if venv exists
if (-not (Test-Path "venv")) {
    Write-Host "Error: Virtual environment not found. Run install.ps1 first." -ForegroundColor Red
    exit 1
}

# Activate venv
.\venv\Scripts\Activate.ps1

# Start server
Write-Host "Starting Meeting Transcriber..."
python -m app.main $args
