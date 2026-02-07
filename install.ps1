# install.ps1 - Meeting Transcriber installation script for Windows
# Run with: powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Meeting Transcriber - Installation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check Python version
Write-Host ""
Write-Host "Checking Python version..."
try {
    $pythonVersion = python --version 2>&1
    $versionMatch = $pythonVersion -match 'Python (\d+)\.(\d+)'
    if ($versionMatch) {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Host "Error: Python 3.10+ required. Found: $pythonVersion" -ForegroundColor Red
            exit 1
        }
        Write-Host "✓ $pythonVersion" -ForegroundColor Green
    }
} catch {
    Write-Host "Error: Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Check NVIDIA GPU
Write-Host ""
Write-Host "Checking GPU..."
$recommendedDtype = "float32"
$recommendedModel = "scerz/VibeVoice-ASR-4bit"

try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Found GPU: $gpuInfo" -ForegroundColor Green

        # Parse compute capability
        if ($gpuInfo -match '(\d+)\.(\d+)$') {
            $computeMajor = [int]$Matches[1]
            if ($computeMajor -ge 8) {
                $recommendedDtype = "bfloat16"
                $recommendedModel = "microsoft/VibeVoice-ASR"
                Write-Host "  Ampere+ GPU detected - will use BF16"
            } else {
                $recommendedDtype = "float16"
                Write-Host "  Pre-Ampere GPU detected - will use FP16 with 4-bit model"
            }
        }
    }
} catch {
    Write-Host "Warning: nvidia-smi not found. GPU support may not work." -ForegroundColor Yellow
}

# Check FFmpeg
Write-Host ""
Write-Host "Checking FFmpeg..."
try {
    $ffmpegVersion = ffmpeg -version 2>&1 | Select-Object -First 1
    Write-Host "✓ FFmpeg found" -ForegroundColor Green
} catch {
    Write-Host "Warning: FFmpeg not found. Please install from https://ffmpeg.org/download.html" -ForegroundColor Yellow
    Write-Host "Add FFmpeg to your PATH after installation." -ForegroundColor Yellow
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..."
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists. Recreating..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}
python -m venv venv
.\venv\Scripts\Activate.ps1
Write-Host "✓ Virtual environment created" -ForegroundColor Green

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA
Write-Host ""
Write-Host "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
Write-Host ""
Write-Host "Installing dependencies..."
pip install -r requirements.txt

# Clone and install VibeVoice
Write-Host ""
Write-Host "Installing VibeVoice..."
if (-not (Test-Path "VibeVoice")) {
    git clone https://github.com/microsoft/VibeVoice.git
}
Push-Location VibeVoice
pip install -e .
Pop-Location

# Create config if not exists
Write-Host ""
Write-Host "Creating configuration..."
if (-not (Test-Path "config.yaml")) {
    @"
server:
  host: "0.0.0.0"
  port: 7860

model:
  path: "$recommendedModel"
  dtype: "$recommendedDtype"
  cache_dir: "./models"
  attn_implementation: "sdpa"

transcription:
  max_file_size_mb: 500
  timeout_seconds: 1800
  default_max_new_tokens: 8192
"@ | Out-File -FilePath "config.yaml" -Encoding UTF8
    Write-Host "✓ Created config.yaml with recommended settings" -ForegroundColor Green
} else {
    Write-Host "config.yaml already exists, skipping" -ForegroundColor Yellow
}

# Create models directory
New-Item -ItemType Directory -Force -Path "models" | Out-Null

# Verify installation
Write-Host ""
Write-Host "Verifying installation..."
python -c @"
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"@

# Verify VibeVoice installation
Write-Host "Verifying VibeVoice installation..." -ForegroundColor Cyan
Set-Location VibeVoice

pip install -e . 2>&1 | Tee-Object -FilePath install.log

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: VibeVoice installation failed. See install.log" -ForegroundColor Red
    exit 1
}

# Verify imports work
python -c @"
import sys
try:
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    print('✓ VibeVoice installed and imports work')
except ImportError as e:
    print(f'✗ VibeVoice import failed: {e}', file=sys.stderr)
    sys.exit(1)
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: VibeVoice imports failed" -ForegroundColor Red
    exit 1
}

Set-Location ..

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start the server:"
Write-Host "  .\run.ps1"
Write-Host ""
Write-Host "Or manually:"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host "  python -m app.main"
Write-Host ""
