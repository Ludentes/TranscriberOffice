# app/ui.py
"""Gradio web interface for transcription."""
import json
import re
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Generator
import uuid

import gradio as gr

from app.transcribe import get_transcription_service
from app.job_service import JobConflictError, JobService, JobValidationError
from app.job_store import JobRecord, JobStatus, JobStore
from app.transcription_queue import transcription_queue


_stop_events: dict[str, threading.Event] = {}
_session_tickets = {}
_stop_events_lock = threading.Lock()


def _get_stop_event(session_id: str) -> threading.Event:
    """Return the cancellation event belonging to one browser session."""
    with _stop_events_lock:
        return _stop_events.setdefault(session_id, threading.Event())


def process_audio_stream(
    audio_path: Optional[str],
    hotwords: str,
    stop_event: Optional[threading.Event] = None,
    use_queue: bool = True,
) -> Generator[tuple[str, str], None, None]:
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
        queue_context = transcription_queue.slot() if use_queue else nullcontext()
        with queue_context:
            if stop_event is not None and stop_event.is_set():
                yield "Stopped by user.", json.dumps(
                    {"success": False, "error": "Stopped"},
                    indent=2,
                    ensure_ascii=False,
                )
                return

            service = get_transcription_service()

            # Use streaming transcription
            for partial_text, final_result in service.transcribe_stream(
                audio_path=audio_path,
                hotwords=hotwords if hotwords else None,
                stop_event=stop_event,
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


def _waiting_status(ticket) -> str:
    """Build a live status for the owner of a queued ticket."""
    snapshot = transcription_queue.snapshot(ticket)
    return (
        f"⏳ **Очередь:** всего задач — {snapshot.total_jobs}. "
        f"Перед вами — {snapshot.position}."
    )


def _global_queue_status() -> str:
    """Build the current global queue status for page visitors."""
    snapshot = transcription_queue.global_snapshot()
    if snapshot.total_jobs == 0:
        return "✅ **Очередь свободна.** Можно запускать транскрибацию."
    return (
        f"⏳ **Сервер занят.** Всего задач: {snapshot.total_jobs}; "
        f"ожидают запуска: {snapshot.waiting_count}."
    )


def _refresh_queue_status(session_id: str) -> str:
    """Refresh the caller's position, or show the global queue."""
    with _stop_events_lock:
        ticket = _session_tickets.get(session_id)
    if ticket is None:
        return _global_queue_status()
    snapshot = transcription_queue.snapshot(ticket)
    if snapshot.is_active:
        return (
            "🎙️ **Ваша транскрибация выполняется сейчас.** "
            f"После вас ожидают: {snapshot.waiting_count}."
        )
    return _waiting_status(ticket)


def extract_text_only(transcript: str) -> str:
    """Strip speaker tags, timestamps, and quotes — return just the spoken text.

    Handles both:
    - Final formatted transcript: [Speaker 1] 00:00:05 - 00:00:12 / "text"
    - Raw streaming JSON: [{"Start":0,"End":39,"Speaker":0,"Content":"text"}, ...]
    """
    if not transcript:
        return ""

    text = transcript.strip()

    # Strip leading status line like "--- Generating (594 tokens, 15.3s) ---"
    if text.startswith("---"):
        newline = text.find("\n")
        if newline != -1:
            text = text[newline + 1:].strip()

    # Strip "assistant" prefix from chat template
    if text.startswith("assistant"):
        text = text[len("assistant"):].strip()

    # Extract Content values from JSON (works on incomplete/streaming JSON too)
    if '"Content"' in text:
        contents = re.findall(r'"Content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if contents:
            return "\n".join(contents)

    # Final formatted transcript: skip [Speaker] and --- lines, strip quotes
    if "[Speaker" in text or "[speaker" in text.lower():
        lines = text.split("\n")
        text_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith("[") or line.startswith("---") or not line:
                continue
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            if line:
                text_lines.append(line)
        return "\n".join(text_lines)

    # Fallback: return as-is (minus status/assistant prefix already stripped)
    return text


_STATUS_LABELS = {
    JobStatus.QUEUED: "⏳ в очереди",
    JobStatus.RUNNING: "🎙️ выполняется",
    JobStatus.COMPLETED: "✅ готово",
    JobStatus.FAILED: "❌ ошибка",
    JobStatus.CANCELED: "⏹ отменено",
}


def build_history_choices(jobs: list[JobRecord]) -> list[tuple[str, str]]:
    """Build safe dropdown labels from an already owner-scoped job list."""
    return [
        (
            f"{_STATUS_LABELS[job.status]} · {job.original_filename} · "
            f"{job.created_at[:16].replace('T', ' ')}",
            job.id,
        )
        for job in jobs
    ]


def choose_history_job(jobs: list[JobRecord], selected_id: str | None) -> str | None:
    job_ids = {job.id for job in jobs}
    if selected_id in job_ids:
        return selected_id
    for job in jobs:
        if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            return job.id
    return jobs[0].id if jobs else None


def build_download_links(job: JobRecord | None) -> str:
    if job is None or job.status != JobStatus.COMPLETED:
        return ""
    return (
        f"[⬇️ Скачать TXT](/api/jobs/{job.id}/download.txt)"
        f" &nbsp;&nbsp; [⬇️ Скачать JSON](/api/jobs/{job.id}/download.json)"
    )


def _owner_from_gradio_request(request: gr.Request) -> str:
    owner_id = getattr(getattr(request, "state", None), "owner_id", None)
    if not owner_id:
        raise gr.Error("Не удалось определить сессию браузера. Обновите страницу.")
    return owner_id


def _job_status_text(job: JobRecord | None, store: JobStore) -> str:
    counts = store.queue_counts()
    queue_suffix = (
        "Очередь свободна."
        if counts.total == 0
        else f"Всего задач: {counts.total}; ожидают: {counts.waiting}."
    )
    if job is None:
        return f"✅ **{queue_suffix}**"
    status = _STATUS_LABELS[job.status]
    if job.error_message:
        return f"**{status}:** {job.error_message}\n\n{queue_suffix}"
    return f"**{status}:** {job.original_filename}\n\n{queue_suffix}"


def _persistent_view(
    store: JobStore,
    owner_id: str,
    selected_id: str | None,
):
    jobs = store.list_jobs(owner_id)
    selected_id = choose_history_job(jobs, selected_id)
    job = store.get_job(owner_id, selected_id) if selected_id else None
    transcript = ""
    json_text = "{}"
    if job is not None:
        transcript = job.result_text or job.progress_text
        json_text = job.result_json or json.dumps(
            {"status": job.status.value, "error": job.error_message},
            ensure_ascii=False,
            indent=2,
        )
    return (
        gr.update(choices=build_history_choices(jobs), value=selected_id),
        transcript,
        json_text,
        _job_status_text(job, store),
        build_download_links(job),
    )


def _create_persistent_ui(store: JobStore, job_service: JobService) -> gr.Blocks:
    """Create the reload-safe UI backed by SQLite jobs."""
    with gr.Blocks(title="Meeting Transcriber") as demo:
        gr.Markdown("# Meeting Transcriber")
        gr.Markdown(
            "Задача продолжится после закрытия вкладки. "
            "История доступна только в этом браузере."
        )
        status_output = gr.Markdown("🔄 Загружаем историю…")
        refresh_timer = gr.Timer(value=2.0, active=True)

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Аудиофайл",
                    type="filepath",
                    sources=["upload"],
                )
                hotwords_input = gr.Textbox(
                    label="Имена и термины (необязательно)",
                    placeholder="ProjectX, Иван Иванов",
                )
                with gr.Row():
                    submit_btn = gr.Button("Транскрибировать", variant="primary")
                    stop_btn = gr.Button("Остановить", variant="stop")

                gr.Markdown("### История")
                history = gr.Dropdown(label="Ваши транскрипции", choices=[])
                with gr.Row():
                    refresh_btn = gr.Button("Обновить", size="sm")
                    delete_btn = gr.Button("Удалить из истории", size="sm")

            with gr.Column(scale=2):
                with gr.Tab("Транскрипция"):
                    text_output = gr.Textbox(label="Текст", lines=20, interactive=False)
                    with gr.Row():
                        snapshot_btn = gr.Button("Подготовить весь текст для копирования", size="sm")
                        snapshot_text_btn = gr.Button("Только речь", size="sm")
                    snapshot_output = gr.Textbox(
                        label="Копирование",
                        lines=12,
                        visible=False,
                        interactive=False,
                    )
                    download_links = gr.Markdown("")
                with gr.Tab("JSON"):
                    json_output = gr.Code(label="JSON", language="json", lines=20)

        def refresh(selected_id, request: gr.Request):
            return _persistent_view(store, _owner_from_gradio_request(request), selected_id)

        def submit(audio_path, hotwords, request: gr.Request):
            owner_id = _owner_from_gradio_request(request)
            if not audio_path:
                view = list(_persistent_view(store, owner_id, None))
                view[3] = "⚠️ **Сначала загрузите аудиофайл.**"
                return tuple(view)
            try:
                job = job_service.create_job(
                    owner_id,
                    audio_path,
                    Path(audio_path).name,
                    hotwords or "",
                )
            except JobValidationError as error:
                view = list(_persistent_view(store, owner_id, None))
                view[3] = f"⚠️ **{error}**"
                return tuple(view)
            return _persistent_view(store, owner_id, job.id)

        def stop(selected_id, request: gr.Request):
            owner_id = _owner_from_gradio_request(request)
            job = store.get_job(owner_id, selected_id) if selected_id else None
            if job and store.request_cancel(owner_id, job.id):
                refreshed = store.get_job(owner_id, job.id)
                if refreshed and refreshed.status == JobStatus.CANCELED:
                    job_service.cleanup_audio(job.id)
            return _persistent_view(store, owner_id, selected_id)

        def delete(selected_id, request: gr.Request):
            owner_id = _owner_from_gradio_request(request)
            try:
                if selected_id:
                    job_service.delete_job(owner_id, selected_id)
            except JobConflictError as error:
                view = list(_persistent_view(store, owner_id, selected_id))
                view[3] = f"⚠️ **{error}**"
                return tuple(view)
            return _persistent_view(store, owner_id, None)

        outputs = [history, text_output, json_output, status_output, download_links]
        demo.load(refresh, inputs=[history], outputs=outputs, queue=False)
        refresh_btn.click(refresh, inputs=[history], outputs=outputs, queue=False)
        refresh_timer.tick(refresh, inputs=[history], outputs=outputs, queue=False)
        history.input(refresh, inputs=[history], outputs=outputs, queue=False)
        submit_btn.click(
            submit,
            inputs=[audio_input, hotwords_input],
            outputs=outputs,
            concurrency_limit=None,
        )
        stop_btn.click(stop, inputs=[history], outputs=outputs, queue=False)
        delete_btn.click(delete, inputs=[history], outputs=outputs, queue=False)
        snapshot_btn.click(
            lambda transcript: gr.update(value=transcript, visible=True),
            inputs=[text_output],
            outputs=[snapshot_output],
        )
        snapshot_text_btn.click(
            lambda transcript: gr.update(value=extract_text_only(transcript), visible=True),
            inputs=[text_output],
            outputs=[snapshot_output],
        )

    return demo


def create_ui(store=None, job_service=None) -> gr.Blocks:
    """Create the Gradio interface."""

    if store is not None and job_service is not None:
        return _create_persistent_ui(store, job_service)

    with gr.Blocks(title="Meeting Transcriber") as demo:
        saved_transcript = gr.State("")
        saved_json = gr.State("")
        session_id = gr.State(lambda: uuid.uuid4().hex)
        gr.Markdown("# Meeting Transcriber")
        gr.Markdown("Upload an MP3 file to transcribe with speaker identification and timestamps.")
        with gr.Row():
            queue_status = gr.Markdown("🔄 Получаем состояние очереди…")
            refresh_queue_btn = gr.Button("Обновить", size="sm", scale=0)
        queue_timer = gr.Timer(value=2.0, active=True)

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
                        label="Transcription (live)",
                        lines=20
                    )
                    with gr.Row():
                        snapshot_btn = gr.Button("Snapshot Full Transcript", size="sm")
                        snapshot_text_btn = gr.Button("Snapshot Text Only", size="sm")
                    snapshot_output = gr.Textbox(
                        label="Snapshot (select and copy from here)",
                        lines=15,
                        visible=False,
                        interactive=False
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

        def stop_transcription(transcript, current_session_id):
            """Stop the current transcription without erasing visible text."""
            _get_stop_event(current_session_id).set()
            return transcript, "⏹ Останавливаем вашу задачу…"

        def start_transcription(
            audio, hotwords, previous_transcript, previous_json, current_session_id
        ):
            """Start transcription and retain every update in this browser session."""
            stop_event = _get_stop_event(current_session_id)
            stop_event.clear()
            ticket = None
            acquired = False
            try:
                yielded = False
                latest_transcript = previous_transcript

                if not audio:
                    for transcript, json_result in process_audio_stream(audio, hotwords):
                        yield (
                            transcript,
                            json_result,
                            transcript,
                            json_result,
                            "⚠️ Сначала загрузите аудиофайл.",
                        )
                    return

                ticket = transcription_queue.enqueue()
                with _stop_events_lock:
                    _session_tickets[current_session_id] = ticket
                while not acquired:
                    if stop_event.is_set():
                        transcription_queue.cancel(ticket)
                        yield (
                            latest_transcript,
                            previous_json,
                            latest_transcript,
                            previous_json,
                            "⏹ Задача удалена из очереди.",
                        )
                        return
                    acquired = transcription_queue.wait(ticket, timeout=1.0)
                    if not acquired:
                        yield (
                            latest_transcript,
                            previous_json,
                            latest_transcript,
                            previous_json,
                            _waiting_status(ticket),
                        )

                active_snapshot = transcription_queue.snapshot(ticket)
                active_status = (
                    "🎙️ **Транскрибация выполняется сейчас.** "
                    f"После вас ожидают: {active_snapshot.waiting_count}."
                )
                yield (
                    latest_transcript,
                    previous_json,
                    latest_transcript,
                    previous_json,
                    active_status,
                )

                for transcript, json_result in process_audio_stream(
                    audio, hotwords, stop_event, use_queue=False
                ):
                    yielded = True
                    try:
                        stopped = json.loads(json_result).get("error") == "Stopped"
                    except (json.JSONDecodeError, AttributeError):
                        stopped = False
                    if stopped and latest_transcript:
                        transcript = latest_transcript
                    else:
                        latest_transcript = transcript
                    yield (
                        transcript,
                        json_result,
                        transcript,
                        json_result,
                        _refresh_queue_status(current_session_id),
                    )
                if not yielded:
                    yield (
                        previous_transcript,
                        previous_json,
                        previous_transcript,
                        previous_json,
                        active_status,
                    )
            finally:
                if ticket is not None:
                    if acquired:
                        transcription_queue.release(ticket)
                    else:
                        transcription_queue.cancel(ticket)
                with _stop_events_lock:
                    _stop_events.pop(current_session_id, None)
                    _session_tickets.pop(current_session_id, None)

            remaining = transcription_queue.snapshot(ticket).total_jobs
            final_status = (
                "✅ **Готово. Очередь свободна.**"
                if remaining == 0
                else f"✅ **Готово.** В общей очереди осталось задач: {remaining}."
            )
            yield (
                latest_transcript,
                previous_json if not yielded else json_result,
                latest_transcript,
                previous_json if not yielded else json_result,
                final_status,
            )

        def snapshot_full(transcript):
            """Snapshot current transcript to a non-streaming textbox."""
            return gr.update(value=transcript, visible=True)

        def snapshot_text(transcript):
            """Snapshot text-only version to a non-streaming textbox."""
            cleaned = extract_text_only(transcript)
            return gr.update(value=cleaned, visible=True)

        # Snapshot buttons — freeze current text into a copyable textbox
        snapshot_btn.click(
            fn=snapshot_full,
            inputs=[text_output],
            outputs=[snapshot_output],
        )
        snapshot_text_btn.click(
            fn=snapshot_text,
            inputs=[text_output],
            outputs=[snapshot_output],
        )

        # Connect the buttons
        stop_btn.click(
            fn=stop_transcription,
            inputs=[saved_transcript, session_id],
            outputs=[text_output, queue_status],
        )
        refresh_queue_btn.click(
            fn=_refresh_queue_status,
            inputs=[session_id],
            outputs=[queue_status],
            queue=False,
        )
        queue_timer.tick(
            fn=_refresh_queue_status,
            inputs=[session_id],
            outputs=[queue_status],
            queue=False,
        )
        transcribe_btn.click(
            fn=start_transcription,
            inputs=[
                audio_input,
                hotwords_input,
                saved_transcript,
                saved_json,
                session_id,
            ],
            outputs=[
                text_output,
                json_output,
                saved_transcript,
                saved_json,
                queue_status,
            ],
            concurrency_limit=None,  # The shared FIFO queue controls GPU access.
        )
        demo.load(
            fn=_refresh_queue_status,
            inputs=[session_id],
            outputs=[queue_status],
            queue=False,
        )

    return demo
