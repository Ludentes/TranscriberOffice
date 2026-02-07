# Reliability and Memory Management Improvements

## Overview

Address 5 critical issues with the Meeting Transcriber: model switching confusion, installation flakiness, memory exhaustion on large files, silent failures, and inability to stop processing.

## Problems Identified

1. **Model switching confusion** - Switching use_quantized setting doesn't unload old model, causing memory issues
2. **Flaky VibeVoice installation** - Package not recognized; AutoModel fallback masks real issues
3. **Memory exhaustion** - Large audio files (>5 min) cause OOM on 24GB GPU
4. **Silent failures** - Users don't see errors during processing
5. **No stop button** - Can't cancel long-running transcriptions

## Solution 1: Audio Chunking for Memory Management

**Strategy:**
- Check audio duration before processing with `librosa.get_duration()`
- If > 5 minutes: split into 3-minute chunks with 10-second overlap
- Process chunks sequentially
- Merge transcriptions with adjusted timestamps
- Clean up temp chunk files

**Implementation:**

```python
def split_audio(audio_path: str, chunk_minutes: int = 3) -> list[str]:
    """Split audio into chunks using ffmpeg."""
    duration = librosa.get_duration(path=audio_path)
    chunk_seconds = chunk_minutes * 60
    overlap_seconds = 10

    chunks = []
    start_time = 0
    chunk_index = 0

    while start_time < duration:
        chunk_path = os.path.join(tempfile.gettempdir(), f"chunk_{chunk_index}.mp3")

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
```

**Merging:**
- Concatenate segment lists from each chunk
- Adjust timestamps: chunk N starts at `N * (chunk_minutes - overlap_minutes)`
- Remove duplicate segments in overlap regions (compare last segment of chunk N with first of chunk N+1)

**Cleanup:**
```python
finally:
    for chunk_path in chunks:
        Path(chunk_path).unlink(missing_ok=True)
```

## Solution 2: Model Switching and Memory Cleanup

**Track loaded model:**
```python
class TranscriptionService:
    def __init__(self, model_path, ...):
        self.model_path = model_path
        self.current_model_path = None  # Track what's loaded
```

**Detect switches and unload:**
```python
def load_model(self):
    # Detect model path change
    if self._loaded and hasattr(self, 'current_model_path'):
        if self.current_model_path != self.model_path:
            print(f"Switching model: {self.current_model_path} -> {self.model_path}")
            self.unload_model()

    if self._loaded:
        return

    print(f"Loading model: {self.model_path}")
    print(f"Device: {self.device}, dtype: {self.dtype}")

    # Existing VibeVoice import and load
    # from_pretrained() handles download automatically

    self.current_model_path = self.model_path

def unload_model(self):
    """Free GPU memory."""
    if self.model is not None:
        del self.model
        del self.processor
    self.model = None
    self.processor = None
    self._loaded = False

    import gc
    gc.collect()
    torch.cuda.empty_cache()
```

**Remove AutoModel fallback:**
- Delete lines 142-156 in app/transcribe.py
- Raise clear ImportError instead:

```python
except ImportError as e:
    raise ImportError(
        f"VibeVoice package not installed.\n"
        f"Please run: ./install.sh\n"
        f"Error: {e}"
    )
```

## Solution 3: Verbose Error Handling

**Stream progress and errors to UI:**

```python
def transcribe_stream(self, audio_path, hotwords, max_new_tokens):
    try:
        duration = librosa.get_duration(path=audio_path)

        if duration > 5 * 60:  # > 5 minutes
            num_chunks = int(duration / (3 * 60)) + 1
            yield f"Audio is {duration/60:.1f} min. Splitting into {num_chunks} chunks...", None

            chunks = split_audio(audio_path)

            for i, chunk_path in enumerate(chunks):
                yield f"Processing chunk {i+1}/{num_chunks}...", None
                # Process chunk
        else:
            yield "Processing audio...", None
            # Single file processing

    except Exception as e:
        error_msg = f"ERROR: {type(e).__name__}: {str(e)}"
        print(error_msg)  # Console logging
        import traceback
        traceback.print_exc()

        yield error_msg, TranscriptionResult(success=False, error=str(e), ...)
```

**All progress messages print to both UI and console.**

## Solution 4: Stop Button

**Global stop flag:**
```python
# Module level in app/transcribe.py
_stop_flag = False

def set_stop_flag(value: bool):
    global _stop_flag
    _stop_flag = value

def check_stop_flag() -> bool:
    global _stop_flag
    return _stop_flag
```

**Check flag during processing:**
```python
def transcribe_stream(self, ...):
    # Before each chunk
    if check_stop_flag():
        yield "Stopped by user.", TranscriptionResult(success=False, error="Stopped")
        return

    # In streaming loop
    for new_text in streamer:
        if check_stop_flag():
            yield "Stopped by user.", TranscriptionResult(success=False, error="Stopped")
            return
```

**UI - Add button:**
```python
# In create_ui()
with gr.Row():
    transcribe_btn = gr.Button("Transcribe", variant="primary")
    stop_btn = gr.Button("Stop", variant="stop")

def stop_transcription():
    from app.transcribe import set_stop_flag
    set_stop_flag(True)
    return "Stopping..."

def start_transcription(audio, hotwords):
    set_stop_flag(False)  # Reset
    return process_audio_stream(audio, hotwords)

stop_btn.click(fn=stop_transcription, outputs=text_output)
transcribe_btn.click(fn=start_transcription, inputs=[audio_input, hotwords_input], outputs=[text_output, json_output])
```

## Solution 5: Installation Reliability

**Verify VibeVoice installation:**

```bash
# In install.sh after VibeVoice install
cd VibeVoice
pip install -e . 2>&1 | tee install.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: VibeVoice installation failed. See install.log"
    exit 1
fi

# Verify imports
python -c "
import sys
try:
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    print('✓ VibeVoice installed and imports work')
except ImportError as e:
    print(f'✗ VibeVoice import failed: {e}', file=sys.stderr)
    sys.exit(1)
"
cd ..
```

**Quick check in run.sh:**
```bash
python -c "from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor" 2>/dev/null || {
    echo "ERROR: VibeVoice not found. Please run ./install.sh"
    exit 1
}
```

## Implementation Order

1. Remove AutoModel fallback (simple, unblocks other work)
2. Add model unloading and switching detection
3. Add installation verification to scripts
4. Add audio chunking with ffmpeg
5. Add stop button and global flag
6. Add verbose error handling and progress messages

## Testing

- Test model switching between full and 4-bit
- Test chunking with 10-minute audio file
- Test stop button during long transcription
- Test error messages display correctly
- Verify VibeVoice installation succeeds on fresh system
