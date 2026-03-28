"""
FormAI web UI: browser live capture (frame API) and static video annotation.

From the project root, use the existing venv:

  venv\\Scripts\\python.exe -m uvicorn ui_app:app --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000/
"""
import base64
import os
import sys
import threading
import uuid
from pathlib import Path

root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)
if os.path.join(root, "src") not in sys.path:
    sys.path.insert(0, os.path.join(root, "src"))

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from video_processor import LivePipeline, process_video

WEB_DIR = Path(__file__).resolve().parent / "web"
DATA_DIR = Path(__file__).resolve().parent / "web_data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FormAI UI")

_sessions: dict[str, LivePipeline] = {}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(500, "Missing web/index.html")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.post("/api/session")
def create_session(exercise: str = "pushup"):
    ex = exercise.strip().lower()
    if ex not in ("pushup", "squat"):
        raise HTTPException(400, "exercise must be pushup or squat")
    sid = str(uuid.uuid4())
    _sessions[sid] = LivePipeline(ex)
    return {"session_id": sid, "exercise": ex}


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"ok": True}


@app.post("/api/frame")
async def process_frame_api(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    if session_id not in _sessions:
        raise HTTPException(404, "Unknown or expired session; create a new one.")

    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image")

    pipeline = _sessions[session_id]
    out_frame, meta = pipeline.process(frame)

    ok, buf = cv2.imencode(".jpg", out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise HTTPException(500, "Failed to encode frame")

    b64 = base64.b64encode(buf).decode("ascii")
    return {
        "image_b64": b64,
        "reps": meta["reps"],
        "stage": meta["stage"],
        "live_msg": meta["live_msg"],
        "feedback_lines": meta["feedback_lines"],
        "smooth_angle": meta["smooth_angle"],
    }


def _run_static_job(job_id: str, input_path: Path, output_path: Path, exercise: str):
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    try:
        coach_summary = process_video(
            str(input_path), str(output_path), exercise, web_playback=True
        )
        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["output_path"] = str(output_path)
            _jobs[job_id]["summary"] = coach_summary or {}
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)


@app.post("/api/process-static")
async def process_static(
    file: UploadFile = File(...),
    exercise: str = Form("pushup"),
):
    ex = exercise.strip().lower()
    if ex not in ("pushup", "squat"):
        raise HTTPException(400, "exercise must be pushup or squat")

    job_id = str(uuid.uuid4())
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        suffix = ".mp4"

    input_path = UPLOAD_DIR / f"{job_id}_in{suffix}"
    output_path = OUTPUT_DIR / f"{job_id}_annotated.mp4"

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")

    input_path.write_bytes(data)

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "output_path": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_run_static_job,
        args=(job_id, input_path, output_path, ex),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job_id")
    out = {
        "status": job["status"],
        "error": job.get("error"),
    }
    if job["status"] == "done" and job.get("summary") is not None:
        out["summary"] = job["summary"]
    return out


@app.get("/api/output/{job_id}")
def job_output(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Output not ready")
    path = job.get("output_path")
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "Output file missing")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename="formai_annotated.mp4",
    )
