#!/bin/bash
# run.sh - Start the Meeting Transcriber server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run ./install.sh first."
    exit 1
fi

# Quick check for VibeVoice installation
./venv/bin/python -c "from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor" 2>/dev/null || {
    echo "ERROR: VibeVoice not found. Please run ./install.sh"
    exit 1
}

# Check GPU compatibility
./venv/bin/python -c "
import torch
import sys

if torch.cuda.is_available():
    try:
        # Try a simple CUDA operation
        x = torch.randn(10, 10).cuda()
        y = x @ x.t()
    except RuntimeError as e:
        if 'no kernel image is available' in str(e) or 'not compatible' in str(e):
            capability = torch.cuda.get_device_capability(0)
            print(f'ERROR: Your GPU (sm_{capability[0]}{capability[1]}) is not compatible with this PyTorch installation.')
            print(f'Please run ./install.sh to reinstall PyTorch with proper CUDA support.')
            sys.exit(1)
        raise
" || exit 1

# Activate venv
source venv/bin/activate

# Start server
echo "Starting Meeting Transcriber..."
python -m app.main "$@"
