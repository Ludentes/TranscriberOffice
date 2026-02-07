#!/bin/bash
# install.sh - Meeting Transcriber installation script for Linux

set -e

echo "=========================================="
echo "  Meeting Transcriber - Installation"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}Warning: Running as root is not recommended.${NC}"
fi

# Check Python version
echo ""
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
        echo -e "${RED}Error: Python 3.10+ required. Found: $PYTHON_VERSION${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
else
    echo -e "${RED}Error: Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

# Check NVIDIA GPU
echo ""
echo "Checking GPU..."
PYTORCH_CUDA_VERSION="cu124"  # Default to CUDA 12.4
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
    echo -e "${GREEN}✓ Found GPU: $GPU_NAME ($GPU_MEMORY)${NC}"

    # Detect GPU architecture for dtype recommendation and CUDA version
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
    if [ "$COMPUTE_CAP" -ge 120 ]; then
        # Blackwell (RTX 50 series) - needs latest PyTorch with sm_120
        RECOMMENDED_DTYPE="bfloat16"
        RECOMMENDED_MODEL="microsoft/VibeVoice-ASR"
        PYTORCH_CUDA_VERSION="cu124"  # Use latest CUDA
        echo "  Blackwell GPU detected (sm_120) - will use BF16 and CUDA 12.4+"
        echo -e "${YELLOW}  Note: Requires PyTorch with sm_120 support${NC}"
    elif [ "$COMPUTE_CAP" -ge 80 ]; then
        # Ampere/Ada (RTX 30/40 series)
        RECOMMENDED_DTYPE="bfloat16"
        RECOMMENDED_MODEL="microsoft/VibeVoice-ASR"
        PYTORCH_CUDA_VERSION="cu121"
        echo "  Ampere+ GPU detected - will use BF16"
    else
        # Pre-Ampere
        RECOMMENDED_DTYPE="float16"
        RECOMMENDED_MODEL="scerz/VibeVoice-ASR-4bit"
        PYTORCH_CUDA_VERSION="cu121"
        echo "  Pre-Ampere GPU detected - will use FP16 with 4-bit model"
    fi
else
    echo -e "${YELLOW}Warning: nvidia-smi not found. GPU support may not work.${NC}"
    RECOMMENDED_DTYPE="float32"
    RECOMMENDED_MODEL="scerz/VibeVoice-ASR-4bit"
    PYTORCH_CUDA_VERSION="cu121"
fi

# Check FFmpeg
echo ""
echo "Checking FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version | head -n1 | cut -d' ' -f3)
    echo -e "${GREEN}✓ FFmpeg $FFMPEG_VERSION${NC}"
else
    echo -e "${YELLOW}FFmpeg not found. Attempting to install...${NC}"
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y ffmpeg
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm ffmpeg
    else
        echo -e "${RED}Error: Could not install FFmpeg. Please install manually.${NC}"
        exit 1
    fi
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Recreating...${NC}"
    rm -rf venv
fi
python3 -m venv venv
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA
echo ""
echo "Installing PyTorch with CUDA support ($PYTORCH_CUDA_VERSION)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$PYTORCH_CUDA_VERSION

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Clone and install VibeVoice
echo ""
echo "Installing VibeVoice..."
if [ ! -d "VibeVoice" ]; then
    git clone https://github.com/microsoft/VibeVoice.git
fi
cd VibeVoice
pip install -e .
cd ..

# Create config if not exists
echo ""
echo "Creating configuration..."
if [ ! -f "config.yaml" ]; then
    cat > config.yaml << EOF
server:
  host: "0.0.0.0"
  port: 7860

model:
  path: "$RECOMMENDED_MODEL"
  dtype: "$RECOMMENDED_DTYPE"
  cache_dir: "./models"
  attn_implementation: "sdpa"

transcription:
  max_file_size_mb: 500
  timeout_seconds: 1800
  default_max_new_tokens: 8192
EOF
    echo -e "${GREEN}✓ Created config.yaml with recommended settings${NC}"
else
    echo -e "${YELLOW}config.yaml already exists, skipping${NC}"
fi

# Create models directory
mkdir -p models

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')

    # Check GPU compatibility
    capability = torch.cuda.get_device_capability(0)
    cap_str = f'sm_{capability[0]}{capability[1]}'
    print(f'Compute capability: {cap_str}')

    # Test if GPU is actually usable
    try:
        x = torch.randn(10, 10).cuda()
        y = x @ x.t()
        print('✓ GPU is working correctly')
    except RuntimeError as e:
        if 'no kernel image is available' in str(e) or 'not compatible' in str(e):
            print(f'✗ WARNING: GPU {cap_str} not compatible with this PyTorch build')
            print(f'  This PyTorch supports: {\" \".join([f\"sm_{c}\" for c in [50, 60, 70, 75, 80, 86, 90]])}')
            print(f'  For {cap_str} support, you may need PyTorch nightly or a newer release')
        else:
            raise
"

# Verify VibeVoice installation
echo "Verifying VibeVoice installation..."
cd VibeVoice || exit 1

pip install -e . 2>&1 | tee install.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: VibeVoice installation failed. See install.log"
    exit 1
fi

# Verify imports work
python -c "
import sys
try:
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    print('✓ VibeVoice installed and imports work')
except ImportError as e:
    print(f'✗ VibeVoice import failed: {e}', file=sys.stderr)
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: VibeVoice imports failed"
    exit 1
fi

cd ..

echo ""
echo "=========================================="
echo -e "${GREEN}  Installation complete!${NC}"
echo "=========================================="
echo ""
echo "To start the server:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python -m app.main"
echo ""
