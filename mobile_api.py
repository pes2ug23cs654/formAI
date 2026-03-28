from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from video_processor import process_video
from src.exercise_profiles import list_exercise_keys

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
REPORT_DIR = BASE_DIR / "report"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FormAI Mobile API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
app.mount("/report", StaticFiles(directory=str(REPORT_DIR)), name="report")

MAX_UPLOAD_MB = 250
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def _parse_report_file(report_path: Path) -> dict | None:
    try:
        if not report_path.exists() or report_path.stat().st_size <= 0:
            return None
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _job_payload_from_paths(request: Request, job_id: str, report_path: Path, output_path: Path) -> dict | None:
    parsed_report = _parse_report_file(report_path)
    if parsed_report is None:
        return None

    base_url = str(request.base_url).rstrip("/")
    return {
        "job_id": job_id,
        "created_at": int(report_path.stat().st_mtime),
        "exercise": parsed_report.get("exercise", "unknown"),
        "summary": parsed_report.get("summary", {}),
        "output_video_url": f"{base_url}/output/{output_path.name}",
        "report_url": f"{base_url}/report/{report_path.name}",
    }


@app.get("/health")
def health() -> dict[str, str | int]:
    return {
        "status": "ok",
        "max_upload_mb": MAX_UPLOAD_MB,
        "exercise_count": len(list_exercise_keys()),
    }


@app.get("/exercises")
def exercises() -> dict[str, list[str]]:
    return {"items": list_exercise_keys()}


@app.get("/recent-jobs")
def recent_jobs(request: Request, limit: int = 10) -> dict[str, list[dict]]:
    normalized_limit = max(1, min(30, int(limit)))
    report_files = sorted(REPORT_DIR.glob("mobile_*_report.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    jobs: list[dict] = []
    for report_path in report_files[:normalized_limit]:
        name = report_path.name
        if not name.startswith("mobile_") or not name.endswith("_report.json"):
            continue
        job_id = name[len("mobile_") : -len("_report.json")]
        output_path = OUTPUT_DIR / f"mobile_{job_id}_output.mp4"
        payload = _job_payload_from_paths(request, job_id, report_path, output_path)
        if payload is not None:
            jobs.append(payload)

    return {"items": jobs}


@app.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict:
    safe_job_id = "".join(ch for ch in job_id if ch.isdigit())
    if not safe_job_id:
        raise HTTPException(status_code=400, detail="Invalid job id")

    report_path = REPORT_DIR / f"mobile_{safe_job_id}_report.json"
    output_path = OUTPUT_DIR / f"mobile_{safe_job_id}_output.mp4"
    payload = _job_payload_from_paths(request, safe_job_id, report_path, output_path)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return payload


def _validate_input_file(file: UploadFile) -> None:
    filename = file.filename or "input.mp4"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )


@app.post("/analyze")
async def analyze_video(
    request: Request,
    file: UploadFile = File(...),
    exercise: str = Form("pushup"),
    calibration_seconds: int = Form(3),
    confidence_threshold: float = Form(0.50),
) -> JSONResponse:
    _validate_input_file(file)

    suffix = Path(file.filename or "input.mp4").suffix or ".mp4"
    job_id = str(int(time.time() * 1000))

    input_path = INPUT_DIR / f"mobile_{job_id}{suffix}"
    output_path = OUTPUT_DIR / f"mobile_{job_id}_output.mp4"
    report_path = REPORT_DIR / f"mobile_{job_id}_report.json"

    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if input_path.stat().st_size <= 0:
            raise HTTPException(status_code=400, detail="Uploaded file was empty")

        if input_path.stat().st_size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size is {MAX_UPLOAD_MB} MB",
            )

        normalized_exercise = str(exercise).strip().lower()
        if normalized_exercise not in set(list_exercise_keys()):
            raise HTTPException(
                status_code=400,
                detail=f"Unknown exercise '{exercise}'. Try /exercises for valid values.",
            )

        normalized_calibration = max(1, min(15, int(calibration_seconds)))
        normalized_confidence = max(0.30, min(0.90, float(confidence_threshold)))

        report = process_video(
            video_path=str(input_path),
            output_path=str(output_path),
            report_json_path=str(report_path),
            debug=False,
            calibration_seconds=normalized_calibration,
            confidence_threshold=normalized_confidence,
            exercise=normalized_exercise,
        )

        if report_path.exists() and report_path.stat().st_size > 0:
            parsed_report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            parsed_report = report

        base_url = str(request.base_url).rstrip("/")
        return JSONResponse(
            {
                "ok": True,
                "job_id": job_id,
                "created_at": int(time.time()),
                "exercise": normalized_exercise,
                "calibration_seconds": normalized_calibration,
                "confidence_threshold": normalized_confidence,
                "report": parsed_report,
                "output_video_url": f"{base_url}/output/{output_path.name}",
                "report_url": f"{base_url}/report/{report_path.name}",
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        try:
            file.file.close()
        except Exception:
            pass



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
