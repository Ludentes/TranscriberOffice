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

# Quick check for VibeVoice installation
.\venv\Scripts\python.exe -c "from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: VibeVoice not found. Please run .\install.ps1" -ForegroundColor Red
    exit 1
}

# Activate venv
.\venv\Scripts\Activate.ps1

# Start server
Write-Host "Starting Meeting Transcriber..."
python -m app.main $args
