# Runbook: Process a Video/Audio Recording (Power Path)

End-to-end pipeline for turning a video recording (e.g. an OBS capture of a
meeting) into transcribed, postprocessed, and summarized meeting notes.

> **Just want the simple version?** If you only need to transcribe and summarize
> one meeting and you have Claude Code, see
> [`runbook-meetings-for-colleagues.md`](runbook-meetings-for-colleagues.md) —
> it covers the easy web-UI path. This document is the *power path*: the full
> batch pipeline run on a GPU box over SSH, best for long (multi-hour)
> recordings.

## Prerequisites

- ffmpeg installed
- VibeVoice-ASR Transcriber app set up on a GPU box (the office boxes are
  `192.168.87.54` and `192.168.87.25` — see the deployment notes)
- Python environment with project dependencies (the repo's `venv`)
- SSH access to the GPU box

## Quick Start

```bash
./scripts/process_recording.sh /path/to/recording.mkv [meeting_name]
```

The script automates steps 1–4 below. Summarization (step 5) requires manual LLM interaction.

## Step-by-Step

### 1. Find the recording and extract the date

OBS, by default, names captures with this convention (Settings → Output →
Recording, or the filename formatting under Settings → Advanced):
```
YYYY-MM-DD HH-MM-SS.mkv
```

Copy the recording somewhere on the GPU box. The date in the filename is the
meeting date. The script converts it to `YY-MM-DD` format for the meeting folder
(e.g., `2026-03-08` → `26-03-08`).

If the date collides with an existing meeting folder, pass a custom name:
```bash
./scripts/process_recording.sh "/home/newub/recordings/2026-03-08 01-59-12.mkv" 26-03-08-consult
```

### 2. Extract audio with ffmpeg

The script extracts audio as MP3 (VBR quality 2, ~128kbps):
```bash
ffmpeg -i input.mkv -vn -acodec libmp3lame -q:a 2 -y output.mp3
```

- Stereo is preserved (helps with speaker diarization if speakers are on different channels)
- A 4-hour recording produces ~200MB MP3
- Extraction takes ~30 seconds (145x realtime)

Output: `meetings/<name>/<name>.mp3`

### 3. Transcribe

Uses `scripts/batch_transcribe.py` which calls the VibeVoice-ASR pipeline:
```bash
python scripts/batch_transcribe.py meetings/<name>/<name>.mp3
```

- Audio is split on silence, then each chunk is transcribed with speaker diarization
- **Speaker IDs reset per chunk** — the same person may get different IDs across chunks
- Skips files that already have `transcript.txt` (safe to re-run)
- ~4 hours of audio takes roughly 30-60 minutes on GPU

Output:
- `meetings/<name>/raw_transcript.json` — segments with timestamps and speaker IDs
- `meetings/<name>/transcript.txt` — formatted transcript

### 4. Postprocess

```bash
python scripts/postprocess.py
```

Processes all meeting directories that have `transcript.txt` but no `clean.txt`. Operations:
- Strips timestamps and chunk labels
- Removes noise segments (`[Music]`, `[Silence]`, `[Human Sounds]`, etc.)
- Removes single-character junk
- Strips surrounding quotes
- **Merges consecutive same-speaker segments** into single blocks
- Typical size reduction: 16–25%

Use `--force` to regenerate existing `clean.txt` files.

Output: `meetings/<name>/clean.txt`

### 5. Summarize

Manual step — feed `clean.txt` to an LLM (Claude, etc.) with a prompt like:

> Summarize this meeting transcript. The participants are [names and roles].
> Structure the summary as:
> - Participants (with speaker ID mapping if identifiable)
> - Context
> - Topics Discussed (numbered, with subsections)
> - Decisions
> - Action Items

Output: `meetings/<name>/summary.md`

#### Tips for speaker identification
- Speaker IDs are unreliable across chunks but consistent within a chunk
- Use contextual clues: speech patterns, role references, gendered verb forms (in Russian)
- Note: the same person talking a lot will dominate Speaker 0 in most chunks
- For a ready-to-paste Claude Code prompt that turns "Speaker N" into real names,
  see the *Name the speakers* section of
  [`runbook-meetings-for-colleagues.md`](runbook-meetings-for-colleagues.md)

### 6. (Optional) Timeline document

After all meetings are summarized, create a chronological timeline:
```
meetings/timeline.md
```

Synthesize all summaries into phases, key decisions, and current project state.

## File Structure

```
meetings/
├── 26-03-08/
│   ├── 26-03-08.mp3          # Extracted audio
│   ├── raw_transcript.json    # Raw segments with timestamps
│   ├── transcript.txt         # Formatted raw transcript
│   ├── clean.txt              # Postprocessed for LLM consumption
│   └── summary.md             # Meeting summary
├── 26-03-08-consult/          # Same date, different recording
│   └── ...
└── timeline.md                # Cross-meeting timeline
```

## Troubleshooting

- **Transcription hangs**: Check GPU memory with `nvidia-smi`. The pipeline loads models into VRAM.
- **Poor diarization**: Expected for long recordings. Speaker IDs reset per silence-split chunk. Postprocessing merges consecutive same-speaker blocks.
- **Missing audio stream**: Check with `ffprobe -show_streams -select_streams a input.mkv`
- **Re-running after failure**: All steps are idempotent — they skip existing outputs. Use `--force` for postprocessing to regenerate.
