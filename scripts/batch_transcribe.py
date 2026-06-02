#!/usr/bin/env python3
"""Batch transcribe audio files using the existing transcription pipeline."""
import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.transcribe import get_transcription_service, format_transcript


def transcribe_file(audio_path: Path, output_dir: Path) -> bool:
    """Transcribe a single audio file and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Transcribing: {audio_path.name}")
    print(f"Output dir:   {output_dir}")
    print(f"{'='*60}")

    service = get_transcription_service()

    start = time.time()
    last_text = ""

    for partial_text, result in service.transcribe_stream(str(audio_path)):
        if result is None:
            # Progress update
            lines = partial_text.split("\n")
            status = lines[0] if lines else partial_text
            # Only print status lines, not full streaming text
            if status.startswith(("Audio is", "Split into", "Processing chunk", "---")):
                print(f"  {status}")
        else:
            # Final result
            elapsed = time.time() - start

            if not result.success:
                print(f"  FAILED: {result.error}")
                return False

            # Save raw segments as JSON
            json_path = output_dir / "raw_transcript.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "source_file": audio_path.name,
                    "duration_seconds": result.duration_seconds,
                    "speakers_detected": result.speakers_detected,
                    "processing_time_seconds": result.processing_time_seconds,
                    "segments": result.segments,
                }, f, ensure_ascii=False, indent=2)

            # Save formatted transcript
            txt_path = output_dir / "transcript.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result.full_text)

            print(f"  Done in {elapsed:.0f}s | "
                  f"{result.duration_seconds/60:.1f}min audio | "
                  f"{result.speakers_detected} speakers | "
                  f"{len(result.segments)} segments")
            print(f"  Saved: {json_path}")
            print(f"  Saved: {txt_path}")
            return True

    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/batch_transcribe.py <audio_file> [audio_file ...]")
        print("       python scripts/batch_transcribe.py temp/archive-*/archive/*.mp3")
        sys.exit(1)

    files = [Path(f) for f in sys.argv[1:]]
    files = [f for f in files if f.exists()]

    if not files:
        print("No valid files found.")
        sys.exit(1)

    # Sort by filename (date-based names sort chronologically)
    files.sort(key=lambda f: f.stem)

    meetings_dir = project_root / "meetings"
    meetings_dir.mkdir(exist_ok=True)

    print(f"Files to process: {len(files)}")
    for f in files:
        print(f"  {f.stem} ({f.stat().st_size / 1024 / 1024:.0f} MB)")

    succeeded = 0
    failed = 0

    for audio_file in files:
        # Use filename stem as folder name (e.g., "25-11-02")
        output_dir = meetings_dir / audio_file.stem
        if (output_dir / "transcript.txt").exists():
            print(f"\nSkipping {audio_file.stem} — already transcribed")
            succeeded += 1
            continue

        ok = transcribe_file(audio_file, output_dir)
        if ok:
            succeeded += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Done: {succeeded} succeeded, {failed} failed out of {len(files)} files")


if __name__ == "__main__":
    main()
