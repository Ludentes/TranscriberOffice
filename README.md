# Meeting Transcriber

A web application and API for transcribing meetings with speaker identification using Microsoft's [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) model.

## Features

- **Long audio support** — silence-based splitting for files of any length
- **Multi-GPU support** — automatically splits model across multiple GPUs
- **Speaker diarization** — automatic speaker identification (Speaker 1, Speaker 2, etc.)
- **Timestamps** — precise timing for each utterance
- **Hotwords support** — improve recognition of names, technical terms, and jargon
- **Stop button** — cancel transcription mid-process
- **Web UI** — simple Gradio interface for uploading and viewing transcripts
- **REST API** — integrate with n8n, Zapier, or custom workflows
- **Cross-platform** — runs on Linux and Windows with NVIDIA GPU

## Requirements

- **Python 3.10+**
- **NVIDIA GPU** with 24GB+ VRAM (RTX 3090, Tesla P40, etc.)
- **CUDA** toolkit installed
- **FFmpeg** for audio processing

### Supported GPUs

| GPU | VRAM | Model Variant | dtype | Multi-GPU |
|-----|------|---------------|-------|-----------|
| RTX 3090/4090 (Ampere+) | 24GB | microsoft/VibeVoice-ASR | bfloat16 | Optional |
| RTX 5090 (Blackwell) | 32GB | microsoft/VibeVoice-ASR | bfloat16 | Optional |
| 2x Tesla P40 (Pascal) | 2x 24GB | microsoft/VibeVoice-ASR | float16 | Required for full model |
| Tesla P40 (Pascal) | 24GB | scerz/VibeVoice-ASR-4bit | float16 | Not needed |

## Quick Start

### Linux

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Transcriber.git
cd Transcriber

# Run the install script
./install.sh

# Start the server
./run.sh
```

### Windows

```powershell
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Transcriber.git
cd Transcriber

# Run the install script (PowerShell)
powershell -ExecutionPolicy Bypass -File install.ps1

# Start the server
powershell -ExecutionPolicy Bypass -File run.ps1
```

After starting, open your browser to `http://localhost:7860`

## Usage

### Web Interface

1. Navigate to `http://localhost:7860`
2. Upload an MP3 file
3. (Optional) Add hotwords for better recognition
4. Click "Transcribe"
5. View results in the Transcript or JSON tab

### REST API

**Endpoint:** `POST /api/transcribe`

```bash
curl -X POST http://localhost:7860/api/transcribe \
  -F "file=@meeting.mp3" \
  -F "hotwords=ProjectX, John Smith, Q4 OKRs"
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
      "text": "Good morning everyone, let's start the standup."
    }
  ],
  "full_text": "[Speaker 1] 00:00:05 - 00:00:12\n\"Good morning everyone...\""
}
```

### n8n Integration

Use the HTTP Request node:
- **Method:** POST
- **URL:** `http://your-server:7860/api/transcribe`
- **Body Content Type:** Multipart Form Data
- **Body Parameters:**
  - `file`: Binary data from previous node
  - `hotwords`: (optional) comma-separated terms

## Configuration

Edit `config.yaml` to customize:

```yaml
server:
  host: "0.0.0.0"  # Listen on all interfaces
  port: 7860

model:
  path: "microsoft/VibeVoice-ASR"  # or "scerz/VibeVoice-ASR-4bit"
  dtype: "auto"                     # auto-detects based on GPU
  cache_dir: "./models"
  attn_implementation: "sdpa"
  use_quantized: "auto"             # "auto", "true", "false"
  device_map: "auto"                # "auto", "single", or "balanced_low_0"

transcription:
  max_file_size_mb: 500
  timeout_seconds: 1800
  default_max_new_tokens: 8192
  chunk_threshold_minutes: 0        # 0 = auto, or minutes before splitting
  chunk_size_minutes: 0             # 0 = auto, or chunk duration in minutes
  chunk_overlap_seconds: 10         # Overlap (only used when no silence found)
  # Silence-based splitting
  silence_split: true               # Split at natural pauses (recommended)
  silence_noise_db: -30             # dB threshold for silence detection
  silence_min_duration: 0.5         # Minimum silence length in seconds
  silence_search_window: 30         # Search window around target boundary (seconds)
```

### Multi-GPU Setup

When `device_map: "auto"` (default), the app automatically detects multiple GPUs and splits the full-precision model across them using [HuggingFace Accelerate](https://huggingface.co/docs/accelerate). The quantized 4-bit model always runs on a single GPU since it's small enough to fit.

| Setting | Behavior |
|---------|----------|
| `device_map: "auto"` | Multi-GPU if 2+ GPUs and full-precision model; single GPU for quantized |
| `device_map: "single"` | Force single GPU (useful for debugging) |

**Notes for multi-GPU:**
- Audio tokenizers and connectors are kept on the same GPU as the embedding layer
- Language model layers are balanced across GPUs by parameter size
- Chunk sizes are automatically calculated based on per-GPU free memory
- Only one transcription runs at a time (queued for concurrent users)

### Audio Chunking

Long audio files are automatically split into chunks to prevent GPU out-of-memory errors. By default, splitting uses **silence detection** — ffmpeg finds natural pauses in the audio and splits at the nearest silence to each target chunk boundary. This avoids cutting mid-sentence and produces better transcriptions than fixed-interval splitting.

When `silence_split: true` (default):
- ffmpeg `silencedetect` finds all pauses in the audio
- Each target chunk boundary snaps to the nearest silence within a configurable search window
- No overlap is needed since splits happen at natural pauses
- Falls back to fixed-interval splitting if no silences are found

Chunk sizes are auto-calculated based on available GPU memory when set to `0`:

| GPU Setup | Model | Approx Chunk Size |
|-----------|-------|-------------------|
| 1x RTX 3090 (24GB) | Full (fp16) | ~3 min |
| 2x Tesla P40 (48GB) | Full (fp16) | ~7 min |
| 1x Tesla P40 (24GB) | 4-bit | ~9 min |
| 1x RTX 4090 (24GB) | Full (bf16) | ~3 min |

Speaker IDs are chunk-local (e.g., "Speaker 1 (Chunk 1)") since speaker identity cannot be tracked across chunks.

### Performance

- **`torch.compile()`** is applied at model load for faster inference (~20-30% speedup after a one-time ~30s warmup)
- **Audio is pre-downsampled** to 24kHz mono during chunk splitting, so the model processor skips per-chunk resampling
- **`torch.inference_mode()`** is used during generation for reduced overhead

## Project Structure

```
Transcriber/
├── app/
│   ├── __init__.py
│   ├── api.py          # REST API endpoints
│   ├── config.py       # Configuration management
│   ├── main.py         # Application entry point
│   ├── transcribe.py   # Transcription service
│   └── ui.py           # Gradio web interface
├── tests/              # Test suite
├── install.sh          # Linux install script
├── install.ps1         # Windows install script
├── run.sh              # Linux run script
├── run.ps1             # Windows run script
├── config.yaml         # Configuration file
└── requirements.txt    # Python dependencies
```

## Guides & Runbooks

Step-by-step guides for everyday use (in [`docs/`](docs/)):

- [**Meetings: from OBS to summary**](docs/runbook-meetings-for-colleagues.md) —
  the office guide. How to transcribe an OBS recording, summarize it, and replace
  "Speaker 0/1" with real names — mostly copy-paste Claude Code prompts. Start here.
- [**Process a recording (power path)**](docs/runbook-process-recording.md) —
  the full batch pipeline over SSH, best for long multi-hour recordings.
- [**Start the Transcriber on machine 25**](docs/runbook-start-transcriber-windows-25.md) —
  how to start the app at the office Windows box (RTX 3090).

## Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux
# or: .\venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

## Acknowledgments

- [Microsoft VibeVoice-ASR](https://github.com/microsoft/VibeVoice) — the underlying speech recognition model
- [Gradio](https://gradio.app/) — web interface framework
- [FastAPI](https://fastapi.tiangolo.com/) — REST API framework

## License

MIT License — see [LICENSE](LICENSE) for details.
