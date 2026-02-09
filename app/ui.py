# app/ui.py
"""Gradio web interface for transcription."""
import json
from typing import Optional, Generator

import gradio as gr

from app.transcribe import get_transcription_service


def process_audio_stream(audio_path: Optional[str], hotwords: str) -> Generator[tuple[str, str], None, None]:
    """Process uploaded audio with streaming output.

    Args:
        audio_path: Path to uploaded audio file
        hotwords: Comma-separated hotwords

    Yields:
        Tuples of (formatted_text, json_string)
    """
    if not audio_path:
        yield "Please upload an audio file.", json.dumps({"success": False, "error": "No audio file uploaded"}, indent=2, ensure_ascii=False)
        return

    try:
        service = get_transcription_service()

        # Use streaming transcription
        for partial_text, final_result in service.transcribe_stream(
            audio_path=audio_path,
            hotwords=hotwords if hotwords else None
        ):
            if final_result is None:
                # Still generating - show partial output
                yield partial_text, json.dumps({"status": "generating..."}, indent=2, ensure_ascii=False)
            else:
                # Final result
                if not final_result.success:
                    error_msg = f"Transcription failed: {final_result.error or 'Unknown error'}"
                    error_json = json.dumps({
                        "success": False,
                        "error": final_result.error,
                        "processing_time_seconds": round(final_result.processing_time_seconds, 2)
                    }, indent=2, ensure_ascii=False)
                    yield error_msg, error_json
                else:
                    # Add processing time to output
                    output_text = final_result.full_text
                    output_text += f"\n--- Completed in {final_result.processing_time_seconds:.1f}s ---"

                    json_response = {
                        "success": True,
                        "duration_seconds": final_result.duration_seconds,
                        "speakers_detected": final_result.speakers_detected,
                        "processing_time_seconds": round(final_result.processing_time_seconds, 2),
                        "segments": final_result.segments,
                        "full_text": final_result.full_text
                    }
                    yield output_text, json.dumps(json_response, indent=2, ensure_ascii=False)

    except Exception as e:
        error_msg = f"Service error: {str(e)}"
        yield error_msg, json.dumps({"success": False, "error": str(e)}, indent=2, ensure_ascii=False)


def extract_text_only(transcript: str) -> str:
    """Strip speaker tags, timestamps, and quotes — return just the spoken text.

    Only processes final transcript format (lines with [Speaker] headers and quoted text).
    If the text doesn't look like a final transcript, returns a message to wait.
    """
    if not transcript:
        return ""

    # Check if this looks like a final transcript (has [Speaker] lines)
    if "[Speaker" not in transcript and "[speaker" not in transcript.lower():
        return "(Transcription still in progress — wait for it to finish)"

    lines = transcript.split("\n")
    text_lines = []
    for line in lines:
        line = line.strip()
        # Skip speaker/timestamp lines, status lines, and empty lines
        if line.startswith("[") or line.startswith("---") or not line:
            continue
        # Strip surrounding quotes
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        if line:
            text_lines.append(line)
    return "\n".join(text_lines)


def create_ui() -> gr.Blocks:
    """Create the Gradio interface."""

    with gr.Blocks(title="Meeting Transcriber") as demo:
        gr.Markdown("# Meeting Transcriber")
        gr.Markdown("Upload an MP3 file to transcribe with speaker identification and timestamps.")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload Audio",
                    type="filepath",
                    sources=["upload"],
                )

                hotwords_input = gr.Textbox(
                    label="Hotwords (optional)",
                    placeholder="ProjectX, John Smith, Q4 OKRs",
                    info="Comma-separated terms to improve recognition"
                )

                with gr.Row():
                    transcribe_btn = gr.Button("Transcribe", variant="primary")
                    stop_btn = gr.Button("Stop", variant="stop")

            with gr.Column(scale=2):
                with gr.Tab("Transcript"):
                    text_output = gr.Textbox(
                        label="Transcription",
                        lines=20
                    )
                    copy_text_btn = gr.Button("Extract Text Only (no timestamps/speakers)", size="sm")
                    text_only_output = gr.Textbox(
                        label="Text Only (use the select-all and copy, or the copy button if available)",
                        lines=10,
                        visible=False
                    )

                with gr.Tab("JSON"):
                    json_output = gr.Code(
                        label="JSON Output",
                        language="json",
                        lines=20
                    )

        with gr.Row():
            gr.Markdown(
                "**Tip:** For best results, ensure clear audio quality. "
                "Add relevant names and terms as hotwords."
            )

        def stop_transcription():
            """Stop the current transcription."""
            from app.transcribe import set_stop_flag
            set_stop_flag(True)
            return "Stopping..."

        def start_transcription(audio, hotwords):
            """Start transcription and reset stop flag."""
            from app.transcribe import set_stop_flag
            set_stop_flag(False)  # Reset flag
            yield from process_audio_stream(audio, hotwords)

        def show_text_only(transcript):
            """Extract text only and show it."""
            cleaned = extract_text_only(transcript)
            return gr.update(value=cleaned, visible=True)

        # Extract Text Only button
        copy_text_btn.click(
            fn=show_text_only,
            inputs=[text_output],
            outputs=[text_only_output],
        )

        # Connect the buttons
        stop_btn.click(fn=stop_transcription, outputs=text_output)
        transcribe_btn.click(
            fn=start_transcription,
            inputs=[audio_input, hotwords_input],
            outputs=[text_output, json_output],
            concurrency_limit=1,  # Only one transcription at a time (GPU constraint)
        )

    return demo
