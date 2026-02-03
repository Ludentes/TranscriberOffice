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

# Activate venv
source venv/bin/activate

# Start server
echo "Starting Meeting Transcriber..."
python -m app.main "$@"
