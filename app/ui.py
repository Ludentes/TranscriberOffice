# app/ui.py
"""Gradio web interface for transcription."""
import json
from typing import Optional

import gradio as gr

from app.transcribe import get_transcription_service


def process_audio(audio_path: Optional[str], hotwords: str) -> tuple[str, str]:
    """Process uploaded audio and return transcript.

    Args:
        audio_path: Path to uploaded audio file
        hotwords: Comma-separated hotwords

    Returns:
        Tuple of (formatted_text, json_string)
    """
    if not audio_path:
        return "Please upload an audio file.", json.dumps({"success": False, "error": "No audio file uploaded"}, indent=2)

    try:
        service = get_transcription_service()
        result = service.transcribe(
            audio_path=audio_path,
            hotwords=hotwords if hotwords else None
        )
    except Exception as e:
        error_msg = f"Service error: {str(e)}"
        return error_msg, json.dumps({"success": False, "error": str(e)}, indent=2)

    if not result.success:
        error_msg = f"Transcription failed: {result.error or 'Unknown error'}"
        error_json = json.dumps({"success": False, "error": result.error}, indent=2)
        return error_msg, error_json

    # Build JSON response
    json_response = {
        "success": True,
        "duration_seconds": result.duration_seconds,
        "speakers_detected": result.speakers_detected,
        "segments": result.segments,
        "full_text": result.full_text
    }

    return result.full_text, json.dumps(json_response, indent=2)


def create_ui() -> gr.Blocks:
    """Create the Gradio interface."""

    with gr.Blocks(
        title="Meeting Transcriber",
        theme=gr.themes.Soft()
    ) as demo:
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

                transcribe_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=2):
                with gr.Tab("Transcript"):
                    text_output = gr.Textbox(
                        label="Transcription",
                        lines=20,
                        show_copy_button=True
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

        # Connect the button
        transcribe_btn.click(
            fn=process_audio,
            inputs=[audio_input, hotwords_input],
            outputs=[text_output, json_output]
        )

    return demo
