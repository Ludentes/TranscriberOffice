"""Owner-isolated API for persistent transcription jobs and history."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel

from app.job_service import JobConflictError, JobService, JobValidationError
from app.job_store import JobRecord, JobStatus, JobStore


class JobSummary(BaseModel):
    id: str
    original_filename: str
    status: JobStatus
    progress_text: str
    error_message: str | None
    created_at: str
    updated_at: str
    completed_at: str | None


class JobDetail(JobSummary):
    hotwords: str
    result_text: str
    result_json: str
    attempt_count: int


class QueueStatus(BaseModel):
    waiting_count: int
    running_count: int
    total_jobs: int


def _summary(job: JobRecord) -> JobSummary:
    return JobSummary(
        id=job.id,
        original_filename=job.original_filename,
        status=job.status,
        progress_text=job.progress_text,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
    )


def _detail(job: JobRecord) -> JobDetail:
    return JobDetail(
        **_summary(job).model_dump(),
        hotwords=job.hotwords,
        result_text=job.result_text,
        result_json=job.result_json,
        attempt_count=job.attempt_count,
    )


def _owner_id(request: Request) -> str:
    owner_id = getattr(request.state, "owner_id", None)
    if not owner_id:
        raise HTTPException(status_code=500, detail="Browser session is unavailable")
    return owner_id


def create_job_router(store: JobStore, service: JobService) -> APIRouter:
    router = APIRouter(prefix="/api/jobs", tags=["transcription-history"])

    def owned_job(request: Request, job_id: str) -> JobRecord:
        job = store.get_job(_owner_id(request), job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Transcription not found")
        return job

    @router.post("", response_model=JobDetail, status_code=202)
    def create_job(
        request: Request,
        file: UploadFile = File(...),
        hotwords: str = Form(""),
    ) -> JobDetail:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        try:
            file.file.seek(0)
            job = service.create_job_from_stream(
                _owner_id(request), file.file, file.filename, hotwords
            )
        except JobValidationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _detail(job)

    @router.get("", response_model=list[JobSummary])
    def list_jobs(request: Request) -> list[JobSummary]:
        return [_summary(job) for job in store.list_jobs(_owner_id(request))]

    @router.get("/queue", response_model=QueueStatus)
    def queue_status() -> QueueStatus:
        counts = store.queue_counts()
        return QueueStatus(
            waiting_count=counts.waiting,
            running_count=counts.running,
            total_jobs=counts.total,
        )

    @router.get("/{job_id}", response_model=JobDetail)
    def get_job(request: Request, job_id: str) -> JobDetail:
        return _detail(owned_job(request, job_id))

    @router.post("/{job_id}/cancel", response_model=JobDetail)
    def cancel_job(request: Request, job_id: str) -> JobDetail:
        owner_id = _owner_id(request)
        job = owned_job(request, job_id)
        if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            raise HTTPException(status_code=409, detail="Transcription is already finished")
        if not store.request_cancel(owner_id, job_id):
            raise HTTPException(status_code=409, detail="Transcription state changed")
        refreshed = store.get_job(owner_id, job_id)
        assert refreshed is not None
        if refreshed.status == JobStatus.CANCELED:
            service.cleanup_audio(job_id)
            refreshed = store.get_job(owner_id, job_id)
            assert refreshed is not None
        return _detail(refreshed)

    @router.delete("/{job_id}", status_code=204)
    def delete_job(request: Request, job_id: str) -> Response:
        owned_job(request, job_id)
        try:
            deleted = service.delete_job(_owner_id(request), job_id)
        except JobConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        if not deleted:
            raise HTTPException(status_code=404, detail="Transcription not found")
        return Response(status_code=204)

    def completed_job(request: Request, job_id: str) -> JobRecord:
        job = owned_job(request, job_id)
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(status_code=409, detail="Transcription is not completed")
        return job

    @router.get("/{job_id}/download.txt")
    def download_text(request: Request, job_id: str) -> Response:
        job = completed_job(request, job_id)
        filename = service.download_name(job.original_filename, "txt")
        return Response(
            content=job.result_text,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @router.get("/{job_id}/download.json")
    def download_json(request: Request, job_id: str) -> Response:
        job = completed_job(request, job_id)
        filename = service.download_name(job.original_filename, "json")
        return Response(
            content=job.result_json,
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return router
