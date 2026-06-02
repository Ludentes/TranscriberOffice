#!/usr/bin/env bash
# Runbook: Process a video/audio recording into transcribed, postprocessed, and summarized meeting notes.
#
# Usage: ./scripts/process_recording.sh /path/to/recording.mkv [meeting_name]
#
# Steps:
#   1. Extract date from filename (format: "YYYY-MM-DD HH-MM-SS.ext")
#   2. Extract audio as MP3 with ffmpeg
#   3. Transcribe using VibeVoice-ASR pipeline
#   4. Postprocess transcript (strip noise, timestamps, merge speakers)
#   5. (Manual) Summarize with LLM
#
# The meeting_name defaults to YY-MM-DD derived from the filename.
# Output goes to meetings/<meeting_name>/

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="./venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: venv not found. Run ./install.sh first."
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <video_or_audio_file> [meeting_name]"
    echo "  File naming convention: 'YYYY-MM-DD HH-MM-SS.ext'"
    echo "  meeting_name defaults to YY-MM-DD from the filename"
    exit 1
fi

INPUT_FILE="$1"

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: File not found: $INPUT_FILE"
    exit 1
fi

# --- Step 1: Extract date from filename ---
BASENAME=$(basename "$INPUT_FILE")
if [[ "$BASENAME" =~ ^([0-9]{4})-([0-9]{2})-([0-9]{2}) ]]; then
    YEAR="${BASH_REMATCH[1]}"
    MONTH="${BASH_REMATCH[2]}"
    DAY="${BASH_REMATCH[3]}"
    # YY-MM-DD format for folder name
    DEFAULT_NAME="${YEAR:2:2}-${MONTH}-${DAY}"
    echo "==> Date from filename: ${YEAR}-${MONTH}-${DAY}"
else
    echo "WARNING: Cannot extract date from filename: $BASENAME"
    DEFAULT_NAME="${BASENAME%.*}"
fi

MEETING_NAME="${2:-$DEFAULT_NAME}"
MEETING_DIR="meetings/${MEETING_NAME}"
mkdir -p "$MEETING_DIR"

echo "==> Meeting directory: $MEETING_DIR"

# --- Step 2: Extract audio ---
# Name audio after the meeting so batch_transcribe.py creates the right output dir
AUDIO_FILE="${MEETING_DIR}/${MEETING_NAME}.mp3"

if [ -f "$AUDIO_FILE" ]; then
    echo "==> Audio already extracted: $AUDIO_FILE"
else
    echo "==> Extracting audio..."
    ffmpeg -i "$INPUT_FILE" -vn -acodec libmp3lame -q:a 2 -y "$AUDIO_FILE" 2>&1 | tail -3
    echo "==> Audio extracted: $AUDIO_FILE ($(du -h "$AUDIO_FILE" | cut -f1))"
fi

# --- Step 3: Transcribe ---
if [ -f "${MEETING_DIR}/transcript.txt" ]; then
    echo "==> Already transcribed: ${MEETING_DIR}/transcript.txt"
else
    echo "==> Transcribing (this may take a while)..."
    # batch_transcribe.py uses filename stem as output dir name,
    # so feeding it meetings/NAME/NAME.mp3 writes to meetings/NAME/
    $PYTHON scripts/batch_transcribe.py "$AUDIO_FILE"
fi

# --- Step 4: Postprocess ---
if [ -f "${MEETING_DIR}/clean.txt" ]; then
    echo "==> Already postprocessed: ${MEETING_DIR}/clean.txt"
else
    echo "==> Postprocessing..."
    $PYTHON scripts/postprocess.py
fi

# --- Step 5: Summary ---
if [ -f "${MEETING_DIR}/summary.md" ]; then
    echo "==> Summary exists: ${MEETING_DIR}/summary.md"
else
    echo ""
    echo "==> Transcription and postprocessing complete."
    echo "    Clean transcript: ${MEETING_DIR}/clean.txt"
    echo "    To summarize, review clean.txt and create ${MEETING_DIR}/summary.md"
    echo "    (Summarization requires LLM — run manually or with Claude)"
fi

echo ""
echo "==> Done. Files in ${MEETING_DIR}/:"
ls -lh "$MEETING_DIR/"
