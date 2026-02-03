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
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
    echo -e "${GREEN}✓ Found GPU: $GPU_NAME ($GPU_MEMORY)${NC}"

    # Detect GPU architecture for dtype recommendation
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
    if [ "$COMPUTE_CAP" -ge 80 ]; then
        RECOMMENDED_DTYPE="bfloat16"
        RECOMMENDED_MODEL="microsoft/VibeVoice-ASR"
        echo "  Ampere+ GPU detected - will use BF16"
    else
        RECOMMENDED_DTYPE="float16"
        RECOMMENDED_MODEL="scerz/VibeVoice-ASR-4bit"
        echo "  Pre-Ampere GPU detected - will use FP16 with 4-bit model"
    fi
else
    echo -e "${YELLOW}Warning: nvidia-smi not found. GPU support may not work.${NC}"
    RECOMMENDED_DTYPE="float32"
    RECOMMENDED_MODEL="scerz/VibeVoice-ASR-4bit"
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
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

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
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

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
