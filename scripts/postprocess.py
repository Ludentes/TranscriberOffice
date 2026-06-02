#!/usr/bin/env python3
"""Postprocess transcripts: strip timestamps/noise, produce clean text for LLM summarization."""
import re
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
MEETINGS_DIR = project_root / "meetings"

# Noise segments to drop entirely (case-insensitive check via bracket pattern)
NOISE_RE = re.compile(r'^\[(Music|Speech|Noise|Applause|Laughter|Silence|Human Sounds?)\]$', re.IGNORECASE)


def clean_transcript(transcript: str) -> str:
    """Strip timestamps, chunk labels, quotes, noise — keep speaker labels and text.

    Also merges consecutive segments from the same speaker into one block.
    """
    lines = transcript.strip().split("\n")

    # First pass: parse into (speaker, text) pairs
    segments = []
    current_speaker = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Speaker header: [Speaker 0 (Chunk 1)] 00:00:00 - 00:00:34
        m = re.match(r'^\[(.+?)(?:\s*\(Chunk \d+\))?\]\s*\d{2}:\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\d{2}$', line)
        if m:
            current_speaker = m.group(1)
            continue

        # Text line (possibly quoted)
        text = line
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        # Drop noise
        if NOISE_RE.match(text.strip()):
            continue

        # Drop single-character junk (like "C")
        if len(text.strip()) <= 1:
            continue

        if current_speaker:
            segments.append((current_speaker, text))
        else:
            segments.append(("Unknown", text))

    # Second pass: merge consecutive same-speaker segments
    merged = []
    for speaker, text in segments:
        if merged and merged[-1][0] == speaker:
            merged[-1] = (speaker, merged[-1][1] + " " + text)
        else:
            merged.append((speaker, text))

    # Format output
    out = []
    for speaker, text in merged:
        out.append(f"\n[{speaker}]")
        out.append(text)

    result = "\n".join(out).strip()
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result


def process_meeting(meeting_dir: Path, force: bool = False) -> bool:
    """Process a single meeting directory."""
    transcript_path = meeting_dir / "transcript.txt"
    clean_path = meeting_dir / "clean.txt"

    if not transcript_path.exists():
        return False

    if clean_path.exists() and not force:
        print(f"  Skipping {meeting_dir.name} — clean.txt exists (use --force to overwrite)")
        return True

    transcript = transcript_path.read_text(encoding="utf-8")
    clean = clean_transcript(transcript)

    clean_path.write_text(clean, encoding="utf-8")

    # Stats
    orig_chars = len(transcript)
    clean_chars = len(clean)
    reduction = (1 - clean_chars / orig_chars) * 100 if orig_chars else 0
    print(f"  {meeting_dir.name}: {orig_chars:,} -> {clean_chars:,} chars ({reduction:.0f}% reduction)")
    return True


def main():
    force = "--force" in sys.argv

    dirs = sorted(MEETINGS_DIR.iterdir()) if MEETINGS_DIR.exists() else []
    dirs = [d for d in dirs if d.is_dir() and (d / "transcript.txt").exists()]

    if not dirs:
        print("No transcribed meetings found.")
        sys.exit(1)

    print(f"Postprocessing {len(dirs)} meetings...")
    for d in dirs:
        process_meeting(d, force=force)
    print("Done.")


if __name__ == "__main__":
    main()
