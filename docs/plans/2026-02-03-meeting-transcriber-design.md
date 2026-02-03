# Meeting Transcriber Design

## Overview

A basic web application and API for transcribing work meetings using Microsoft's VibeVoice-ASR model. Runs on office network, provides web UI for manual uploads and REST API for n8n integration.

## Model: VibeVoice-ASR

- **Size:** 9B parameters
- **Capabilities:** 60-minute audio in single pass, speaker diarization, timestamps, 50+ languages, custom hotwords
- **Variants:**
  - `microsoft/VibeVoice-ASR` — BF16 (17.3GB) for Ampere+ GPUs
  - `scerz/VibeVoice-ASR-4bit` — 4-bit quantized for older GPUs

## Target Hardware

| Machine | GPU | OS | Model Variant |
|---------|-----|-------|---------------|
| Primary | RTX 3090 (24GB) | Windows | BF16 official |
| Secondary | Tesla P40 (24GB) | Linux | 4-bit quantized |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Python Application                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │              FastAPI Application                 │   │
│  │  ┌───────────────┐    ┌───────────────────┐    │   │
│  │  │  Gradio UI    │    │  /api/transcribe  │    │   │
│  │  │  (mounted)    │    │  (n8n endpoint)   │    │   │
│  │  └───────┬───────┘    └─────────┬─────────┘    │   │
│  │          │                      │               │   │
│  │          └──────────┬───────────┘               │   │
│  │                     ▼                           │   │
│  │           ┌─────────────────┐                   │   │
│  │           │ Transcription   │                   │   │
│  │           │ Service         │                   │   │
│  │           └────────┬────────┘                   │   │
│  └────────────────────┼────────────────────────────┘   │
│                       ▼                                 │
│             ┌─────────────────┐                         │
│             │  VibeVoice-ASR  │                         │
│             │  (HuggingFace)  │                         │
│             └─────────────────┘                         │
└─────────────────────────────────────────────────────────┘
```

## Directory Structure

```
Transcriber/
├── install.sh              # Linux install script
├── install.ps1             # Windows install script
├── run.sh                  # Linux run script
├── run.ps1                 # Windows run script
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI + Gradio mount
│   ├── api.py              # REST endpoints
│   ├── transcribe.py       # Core transcription logic
│   └── ui.py               # Gradio interface
├── venv/                   # Python virtual environment
├── models/                 # Optional local model cache
└── docs/
    └── plans/
```

## Web UI

Gradio-based interface with:

- File upload (drag-and-drop) for MP3 files
- Optional hotwords text field (comma-separated)
- Progress indicator during transcription
- Formatted transcript display with speaker labels and timestamps
- Download buttons for text and JSON formats

## REST API

### `POST /api/transcribe`

**Request:**
```
Content-Type: multipart/form-data

- file: (binary) MP3 file
- hotwords: (optional) comma-separated string
```

**Response:**
```json
{
  "success": true,
  "duration_seconds": 1847,
  "speakers_detected": 3,
  "segments": [
    {
      "speaker": "Speaker 1",
      "start": "00:00:05",
      "end": "00:00:12",
      "text": "Good morning everyone, let's start."
    }
  ],
  "full_text": "[Speaker 1] 00:00:05 - 00:00:12\n\"Good morning everyone...\""
}
```

**Error response:**
```json
{
  "success": false,
  "error": "File format not supported. Expected MP3."
}
```

## Configuration

**`config.yaml`:**
```yaml
server:
  host: "0.0.0.0"
  port: 7860

model:
  path: "microsoft/VibeVoice-ASR"
  dtype: "auto"
  cache_dir: "./models"

transcription:
  max_file_size_mb: 500
  timeout_seconds: 1800
```

## Install Scripts

**Responsibilities:**

1. Check prerequisites:
   - Python 3.10+
   - NVIDIA GPU detected
   - CUDA toolkit available
   - FFmpeg installed

2. Create isolated environment:
   - Python venv in `./venv`
   - Install PyTorch with CUDA
   - Install VibeVoice from GitHub
   - Install app dependencies

3. Configure:
   - Create `config.yaml` with defaults
   - Auto-detect GPU and suggest model variant
   - Set appropriate torch dtype

4. Verify:
   - Quick model load test
   - Report success/failure

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Deployment | Native Python (no Docker) | Easier GPU debugging, simpler setup |
| Input format | MP3 | Covers common meeting recordings |
| Output formats | Text + JSON | Human-readable + machine-processable |
| API style | Synchronous | Simpler for n8n, acceptable for internal use |
| Authentication | None | Trusted office network |
| Speaker labels | Generic (Speaker 1, 2...) | MVP simplicity |
| Hotwords | Per-request text field | Simple, expandable later |
| Framework | FastAPI + Gradio | Python ecosystem, existing VibeVoice Gradio demo |

## Future Enhancements (Out of Scope for MVP)

- Saved hotword lists
- Speaker name mapping
- Additional audio formats
- Async API with webhooks
- User authentication
- Batch processing
