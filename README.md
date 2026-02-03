# Meeting Transcriber

A web application and API for transcribing meetings with speaker identification using Microsoft's [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) model.

## Features

- **60-minute single-pass transcription** — process long meetings without chunking
- **Speaker diarization** — automatic speaker identification (Speaker 1, Speaker 2, etc.)
- **Timestamps** — precise timing for each utterance
- **Hotwords support** — improve recognition of names, technical terms, and jargon
- **Web UI** — simple Gradio interface for uploading and viewing transcripts
- **REST API** — integrate with n8n, Zapier, or custom workflows
- **Cross-platform** — runs on Linux and Windows with NVIDIA GPU

## Requirements

- **Python 3.10+**
- **NVIDIA GPU** with 24GB+ VRAM (RTX 3090, Tesla P40, etc.)
- **CUDA** toolkit installed
- **FFmpeg** for audio processing

### Supported GPUs

| GPU | VRAM | Model Variant | dtype |
|-----|------|---------------|-------|
| RTX 3090/4090 (Ampere+) | 24GB | microsoft/VibeVoice-ASR | bfloat16 |
| Tesla P40 (Pascal) | 24GB | scerz/VibeVoice-ASR-4bit | float16 |

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
  dtype: "auto"  # auto-detects based on GPU
  cache_dir: "./models"
  attn_implementation: "sdpa"

transcription:
  max_file_size_mb: 500
  timeout_seconds: 1800
  default_max_new_tokens: 8192
```

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
