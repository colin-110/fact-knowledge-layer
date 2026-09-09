"""In-process background job runner: a single worker thread pulling from a
queue. This replaces Redis/RQ for the prototype - simple, no extra infra, and
still gives real async processing (upload returns immediately, a job row is
polled for status). Swapping in a real queue later only means changing
`enqueue`.
"""

import queue
import threading

_task_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
_worker_thread: threading.Thread | None = None
_lock = threading.Lock()


def _worker_loop():
    from app import storage
    from app.pipeline.ingestion import run_document_pipeline

    while True:
        document_id, job_id = _task_queue.get()
        try:
            run_document_pipeline(document_id, job_id)
        except Exception as exc:  # noqa: BLE001
            storage.update_job(job_id, status="failed", error=str(exc)[:2000])
        finally:
            _task_queue.task_done()


def start_worker():
    global _worker_thread
    with _lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="ingestion-worker")
            _worker_thread.start()


def enqueue(document_id: str, job_id: str):
    start_worker()
    _task_queue.put((document_id, job_id))
