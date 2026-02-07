# app/transcribe.py
"""Transcription service using VibeVoice-ASR."""
import re
import time
import threading
import subprocess
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Generator

import torch


# Global stop flag for canceling transcription
_stop_flag = False


def set_stop_flag(value: bool) -> None:
    """Set the global stop flag."""
    global _stop_flag
    _stop_flag = value


def check_stop_flag() -> bool:
    """Check the global stop flag."""
    global _stop_flag
    return _stop_flag


@dataclass
class TranscriptionResult:
    """Result from transcription."""
    success: bool
    segments: list[dict]
    full_text: str
    duration_seconds: float
    speakers_detected: int
    processing_time_seconds: float = 0.0
    error: Optional[str] = None


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_transcript(segments: list[dict]) -> str:
    """Format segments into human-readable transcript."""
    lines = []
    for seg in segments:
        start = format_timestamp(seg.get("start", 0))
        end = format_timestamp(seg.get("end", 0))
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        lines.append(f'[{speaker}] {start} - {end}')
        lines.append(f'"{text}"')
        lines.append("")
    return "\n".join(lines)


def parse_model_output(raw_output: str) -> list[dict]:
    """Parse VibeVoice model output into structured segments.

    VibeVoice outputs in format like:
    <|speaker_1|> text <|end|>
    Or with timestamps:
    <|0.00|> <|speaker_1|> text <|end|>
    """
    segments = []

    # Pattern to match speaker segments with optional timestamps
    # This handles various VibeVoice output formats
    pattern = r'(?:<\|(\d+\.?\d*)\|>\s*)?<\|speaker_(\d+)\|>\s*(.*?)\s*<\|end\|>'

    matches = re.findall(pattern, raw_output, re.DOTALL | re.IGNORECASE)

    current_time = 0.0
    for match in matches:
        timestamp_str, speaker_id, text = match

        start_time = float(timestamp_str) if timestamp_str else current_time
        # Estimate end time based on text length (rough: 150 words/min)
        words = len(text.split())
        duration = max(1.0, words / 2.5)  # ~150 wpm
        end_time = start_time + duration
        current_time = end_time

        segments.append({
            "speaker": f"Speaker {speaker_id}",
            "start": start_time,
            "end": end_time,
            "text": text.strip()
        })

    # If no pattern matches, try simpler parsing or return raw
    if not segments and raw_output.strip():
        segments.append({
            "speaker": "Speaker 1",
            "start": 0.0,
            "end": 0.0,
            "text": raw_output.strip()
        })

    return segments


def split_audio(audio_path: str, chunk_minutes: int = 3) -> list[str]:
    """Split audio into chunks using ffmpeg.

    Args:
        audio_path: Path to audio file
        chunk_minutes: Duration of each chunk in minutes

    Returns:
        List of paths to chunk files
    """
    import librosa

    duration = librosa.get_duration(path=audio_path)
    chunk_seconds = chunk_minutes * 60
    overlap_seconds = 10

    chunks = []
    start_time = 0
    chunk_index = 0

    while start_time < duration:
        chunk_path = os.path.join(tempfile.gettempdir(), f"chunk_{os.getpid()}_{chunk_index}.mp3")

        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-t", str(chunk_seconds),
            "-i", audio_path,
            "-c", "copy",  # No re-encoding for speed
            chunk_path
        ], check=True, capture_output=True)

        chunks.append(chunk_path)
        start_time += (chunk_seconds - overlap_seconds)
        chunk_index += 1

    return chunks


class TranscriptionService:
    """Service for transcribing audio using VibeVoice-ASR."""

    def __init__(
        self,
        model_path: str = "microsoft/VibeVoice-ASR",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        attn_implementation: str = "sdpa"
    ):
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.cache_dir = cache_dir
        self.attn_implementation = attn_implementation
        self.model = None
        self.processor = None
        self._loaded = False
        self.current_model_path = None  # Track what's loaded

    def load_model(self) -> None:
        """Load the VibeVoice-ASR model."""
        # Detect model path change
        if self._loaded and self.current_model_path != self.model_path:
            print(f"Switching model: {self.current_model_path} -> {self.model_path}")
            self.unload_model()

        if self._loaded:
            return

        print(f"Loading model: {self.model_path}")
        print(f"Device: {self.device}, dtype: {self.dtype}")

        # Try VibeVoice-specific imports first
        try:
            from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
            from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration

            self.processor = VibeVoiceASRProcessor.from_pretrained(
                self.model_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir,
                attn_implementation=self.attn_implementation,
                trust_remote_code=True
            )
        except ImportError as e:
            raise ImportError(
                f"VibeVoice package not installed.\n"
                f"Please run: ./install.sh (Linux/Mac) or .\\install.ps1 (Windows)\n"
                f"Error: {e}"
            )

        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        self.current_model_path = self.model_path
        print("Model loaded successfully")

    def unload_model(self) -> None:
        """Free GPU memory by unloading model."""
        if self.model is not None:
            del self.model
            del self.processor
        self.model = None
        self.processor = None
        self._loaded = False

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("Model unloaded and GPU memory freed")

    def transcribe(
        self,
        audio_path: str,
        hotwords: Optional[str] = None,
        max_new_tokens: int = 8192
    ) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            hotwords: Optional comma-separated hotwords for better recognition
            max_new_tokens: Maximum tokens to generate

        Returns:
            TranscriptionResult with segments and formatted text
        """
        if not self._loaded:
            self.load_model()

        start_time = time.time()

        try:
            # Build context info from hotwords
            context_info = None
            if hotwords:
                terms = [h.strip() for h in hotwords.split(",") if h.strip()]
                if terms:
                    context_info = f"Key terms: {', '.join(terms)}"

            # Process audio
            inputs = self.processor(
                audio=audio_path,
                return_tensors="pt",
                add_generation_prompt=True,
                context_info=context_info
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate transcription
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    pad_token_id=getattr(self.processor, 'pad_id', self.processor.tokenizer.pad_token_id),
                )

            # Decode output
            generated_text = self.processor.decode(
                output_ids[0],
                skip_special_tokens=False  # Keep speaker tokens for parsing
            )

            # Try to use processor's post_process if available
            try:
                segments = self.processor.post_process_transcription(generated_text)
                # Convert to our format if needed
                if segments and isinstance(segments[0], dict):
                    pass  # Already in dict format
                else:
                    segments = parse_model_output(generated_text)
            except (AttributeError, TypeError, ValueError):
                segments = parse_model_output(generated_text)

            # Get audio duration
            import librosa
            duration = librosa.get_duration(path=audio_path)

            # Count unique speakers
            speakers = set(seg.get("speaker", "") for seg in segments)

            processing_time = time.time() - start_time

            return TranscriptionResult(
                success=True,
                segments=segments,
                full_text=format_transcript(segments),
                duration_seconds=duration,
                speakers_detected=len(speakers),
                processing_time_seconds=processing_time,
                error=None
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return TranscriptionResult(
                success=False,
                segments=[],
                full_text="",
                duration_seconds=0,
                speakers_detected=0,
                processing_time_seconds=processing_time,
                error=str(e)
            )

    def transcribe_stream(
        self,
        audio_path: str,
        hotwords: Optional[str] = None,
        max_new_tokens: int = 8192
    ) -> Generator[tuple[str, Optional[TranscriptionResult]], None, None]:
        """Transcribe an audio file with streaming output.

        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            hotwords: Optional comma-separated hotwords for better recognition
            max_new_tokens: Maximum tokens to generate

        Yields:
            Tuples of (partial_text, final_result) where final_result is None until complete
        """
        if not self._loaded:
            self.load_model()

        start_time = time.time()
        chunks = []

        try:
            import librosa
            from transformers import TextIteratorStreamer

            # Check stop flag before starting
            if check_stop_flag():
                msg = "Stopped by user."
                print(msg)
                yield msg, TranscriptionResult(
                    success=False, segments=[], full_text="",
                    duration_seconds=0, speakers_detected=0, error="Stopped"
                )
                return

            # Check audio duration
            duration = librosa.get_duration(path=audio_path)

            # If audio is long, split into chunks
            if duration > 5 * 60:  # > 5 minutes
                num_chunks = int(duration / (3 * 60)) + 1
                msg = f"Audio is {duration/60:.1f} minutes. Splitting into {num_chunks} chunks..."
                print(msg)  # Console logging
                yield msg, None

                chunks = split_audio(audio_path, chunk_minutes=3)
                all_segments = []

                for i, chunk_path in enumerate(chunks):
                    # Check stop flag before each chunk
                    if check_stop_flag():
                        msg = "Stopped by user."
                        print(msg)
                        yield msg, TranscriptionResult(
                            success=False, segments=[], full_text="",
                            duration_seconds=0, speakers_detected=0, error="Stopped"
                        )
                        return

                    msg = f"Processing chunk {i+1}/{len(chunks)}..."
                    print(msg)  # Console logging
                    yield msg, None

                    # Process this chunk
                    chunk_start_time = i * (3 * 60 - 10)  # Account for overlap

                    for partial_text, partial_result in self._transcribe_single_stream(
                        chunk_path, hotwords, max_new_tokens
                    ):
                        if partial_result is None:
                            yield f"Chunk {i+1}/{len(chunks)}: {partial_text}", None
                        else:
                            # Adjust timestamps and merge segments
                            for seg in partial_result.segments:
                                seg["start"] += chunk_start_time
                                seg["end"] += chunk_start_time
                            all_segments.extend(partial_result.segments)

                # Build final result
                processing_time = time.time() - start_time
                speakers = set(seg.get("speaker", "") for seg in all_segments)

                result = TranscriptionResult(
                    success=True,
                    segments=all_segments,
                    full_text=format_transcript(all_segments),
                    duration_seconds=duration,
                    speakers_detected=len(speakers),
                    processing_time_seconds=processing_time,
                    error=None
                )

                yield result.full_text, result
            else:
                # Single file processing
                msg = "Processing audio..."
                print(msg)  # Console logging
                yield msg, None

                for partial_text, final_result in self._transcribe_single_stream(
                    audio_path, hotwords, max_new_tokens
                ):
                    yield partial_text, final_result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"ERROR: {type(e).__name__}: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()

            result = TranscriptionResult(
                success=False,
                segments=[],
                full_text="",
                duration_seconds=0,
                speakers_detected=0,
                processing_time_seconds=processing_time,
                error=str(e)
            )
            yield error_msg, result
        finally:
            # Clean up chunk files
            for chunk_path in chunks:
                Path(chunk_path).unlink(missing_ok=True)

    def _transcribe_single_stream(
        self,
        audio_path: str,
        hotwords: Optional[str],
        max_new_tokens: int
    ) -> Generator[tuple[str, Optional[TranscriptionResult]], None, None]:
        """Transcribe a single audio file with streaming (internal helper)."""
        from transformers import TextIteratorStreamer
        import librosa

        start_time = time.time()

        # Build context info from hotwords
        context_info = None
        if hotwords:
            terms = [h.strip() for h in hotwords.split(",") if h.strip()]
            if terms:
                context_info = f"Key terms: {', '.join(terms)}"

        # Process audio
        inputs = self.processor(
            audio=audio_path,
            return_tensors="pt",
            add_generation_prompt=True,
            context_info=context_info
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Create streamer
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Container for result from thread
        result_container = {"text": "", "error": None}

        def generate_thread():
            try:
                with torch.no_grad():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        pad_token_id=getattr(self.processor, 'pad_id', self.processor.tokenizer.pad_token_id),
                        streamer=streamer,
                    )
            except Exception as e:
                result_container["error"] = str(e)

        # Start generation in background thread
        thread = threading.Thread(target=generate_thread)
        thread.start()

        # Stream output
        generated_text = ""
        token_count = 0
        for new_text in streamer:
            # Check stop flag during generation
            if check_stop_flag():
                msg = "Stopped by user."
                print(msg)
                yield msg, TranscriptionResult(
                    success=False, segments=[], full_text="",
                    duration_seconds=0, speakers_detected=0, error="Stopped"
                )
                return

            generated_text += new_text
            token_count += 1
            elapsed = time.time() - start_time
            status = f"--- Generating ({token_count} tokens, {elapsed:.1f}s) ---\n{generated_text}"
            yield status, None

        thread.join()

        if result_container["error"]:
            raise Exception(result_container["error"])

        # Get full output with special tokens for parsing
        full_output = self.processor.decode(
            self.processor.tokenizer.encode(generated_text),
            skip_special_tokens=False
        )

        # Parse segments
        try:
            segments = self.processor.post_process_transcription(full_output)
            if not (segments and isinstance(segments[0], dict)):
                segments = parse_model_output(generated_text)
        except (AttributeError, TypeError, ValueError):
            segments = parse_model_output(generated_text)

        # Get audio duration
        duration = librosa.get_duration(path=audio_path)

        # Count unique speakers
        speakers = set(seg.get("speaker", "") for seg in segments)

        processing_time = time.time() - start_time

        result = TranscriptionResult(
            success=True,
            segments=segments,
            full_text=format_transcript(segments),
            duration_seconds=duration,
            speakers_detected=len(speakers),
            processing_time_seconds=processing_time,
            error=None
        )

        yield result.full_text, result


# Global service instance
_service: Optional[TranscriptionService] = None


def get_transcription_service() -> TranscriptionService:
    """Get or create the global transcription service."""
    global _service
    if _service is None:
        from app.config import get_config, get_torch_dtype, get_model_path
        config = get_config()
        _service = TranscriptionService(
            model_path=get_model_path(config.model),
            dtype=get_torch_dtype(config.model.dtype),
            cache_dir=config.model.cache_dir,
            attn_implementation=config.model.attn_implementation
        )
    return _service
