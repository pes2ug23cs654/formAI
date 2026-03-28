"""FormAI Flask UI — same flows as app_streamlit (library, upload, file analysis, live webcam)."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, send_file, url_for

WEB_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WEB_ROOT.parent
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from formai_core import (
    BASE_DIR,
    CANONICAL_OUTPUT,
    CANONICAL_REPORT,
    INPUT_DIR,
    LIVE_SESSION_REPORT,
    MODE_LIBRARY,
    MODE_LIVE,
    MODE_UPLOAD,
    UPLOAD_INPUT,
    build_holistic_feedback,
    get_demo_input_video,
    get_guide_image_for_exercise,
    get_web_preview_video,
    list_input_videos,
    load_report_json,
    run_live_webcam_assessment,
    save_report_json,
)
from src.exercise_profiles import list_exercise_keys
from video_processor import process_video

app = Flask(
    __name__,
    template_folder=str(WEB_ROOT / "templates"),
    static_folder=str(WEB_ROOT / "static"),
)
app.secret_key = "formai-dev-secret-change-in-production"

_live_lock = threading.Lock()
_live_state: dict = {
    "mode": "idle",
    "frame_jpeg": None,
    "stats": {},
    "error": None,
    "report": None,
}


def _reset_live_state_for_new_session() -> None:
    _live_state["frame_jpeg"] = None
    _live_state["stats"] = {}
    _live_state["error"] = None
    _live_state["report"] = None


def _allowed_video_basename(name: str) -> bool:
    allowed = {p.name for p in list_input_videos()}
    return name in allowed


def _index_context(default_mode: str = MODE_LIBRARY, exercise: str | None = None) -> dict:
    keys = list_exercise_keys()
    ex = exercise if exercise in keys else "pushup"
    vids = list_input_videos()
    names = [p.name for p in vids]
    demo_path = get_demo_input_video(vids)
    guide_urls: dict[str, str | None] = {}
    for key in keys:
        g = get_guide_image_for_exercise(key)
        guide_urls[key] = url_for("guide_image", name=g.name) if g is not None else None

    default_video = "input.mp4" if "input.mp4" in names else (names[0] if names else "")

    library_video_urls: dict[str, str] = {n: url_for("serve_input_video", name=n) for n in names}

    video_report = None
    preview_err = None
    video_url = None
    download_url = None
    coaching_tips: list[str] = []
    show_latest = False
    if CANONICAL_OUTPUT.exists() and CANONICAL_REPORT.exists():
        data = load_report_json(CANONICAL_REPORT)
        if data:
            show_latest = True
            video_report = data
            summary = data.get("summary", {}) or {}
            coaching_tips = build_holistic_feedback(data.get("exercise", "-"), summary)
            preview_path, preview_err = get_web_preview_video(CANONICAL_OUTPUT)
            if preview_path.exists():
                video_url = url_for("video_preview")
            download_url = url_for("video_download")

    return {
        "exercise_keys": keys,
        "default_exercise": ex,
        "default_calibration": 3,
        "default_confidence": 0.5,
        "mode_library": MODE_LIBRARY,
        "mode_upload": MODE_UPLOAD,
        "mode_live": MODE_LIVE,
        "default_mode": default_mode,
        "video_names": names,
        "default_video": default_video,
        "demo_available": demo_path is not None,
        "demo_name": demo_path.name if demo_path else "",
        "guide_urls": guide_urls,
        "library_video_urls": library_video_urls,
        "show_latest": show_latest,
        "video_report": video_report,
        "preview_err": preview_err,
        "video_url": video_url,
        "download_url": download_url,
        "coaching_tips": coaching_tips,
    }


@app.get("/")
def index():
    with _live_lock:
        if _live_state["mode"] == "finished":
            _live_state["mode"] = "idle"
    return render_template("index.html", **_index_context())


@app.get("/guide/<name>")
def guide_image(name: str):
    from formai_core import PICS_DIR

    if ".." in name or "/" in name or "\\" in name:
        return ("Not found", 404)
    path = (PICS_DIR / name).resolve()
    if not str(path).startswith(str(PICS_DIR.resolve())) or not path.is_file():
        return ("Not found", 404)
    return send_file(path)


def _mime_for_video_filename(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext == ".mov":
        return "video/quicktime"
    if ext in (".webm",):
        return "video/webm"
    return "video/mp4"


@app.get("/media/input/<name>")
def serve_input_video(name: str):
    if not _allowed_video_basename(name):
        return ("Not found", 404)
    path = INPUT_DIR / name
    return send_file(path, mimetype=_mime_for_video_filename(name))


@app.get("/video/preview")
def video_preview():
    preview_path, _ = get_web_preview_video(CANONICAL_OUTPUT)
    if not preview_path.exists():
        return ("No output yet", 404)
    return send_file(preview_path, mimetype="video/mp4")


@app.get("/video/download")
def video_download():
    preview_path, _ = get_web_preview_video(CANONICAL_OUTPUT)
    if not preview_path.exists():
        return ("No output yet", 404)
    return send_file(preview_path, as_attachment=True, download_name=preview_path.name)


@app.get("/results")
def results():
    if not CANONICAL_REPORT.exists():
        flash("No saved report yet. Run analysis first.", "info")
        return redirect(url_for("index"))
    data = load_report_json(CANONICAL_REPORT)
    if not data:
        flash("Report file is invalid or empty.", "error")
        return redirect(url_for("index"))
    summary = data.get("summary", {}) or {}
    coaching_tips = build_holistic_feedback(data.get("exercise", "-"), summary)
    preview_path, preview_err = get_web_preview_video(CANONICAL_OUTPUT)
    video_url = url_for("video_preview") if preview_path.exists() else None
    download_url = url_for("video_download") if preview_path.exists() else None
    return render_template(
        "results_video.html",
        video_report=data,
        coaching_tips=coaching_tips,
        preview_err=preview_err,
        video_url=video_url,
        download_url=download_url,
    )


@app.get("/live/results")
def live_results():
    data = load_report_json(LIVE_SESSION_REPORT)
    if not data:
        flash("No live session report found. Run a live assessment first.", "info")
        return redirect(url_for("index"))
    summary = data.get("summary", {}) or {}
    coaching_tips = build_holistic_feedback(data.get("exercise", "-"), summary)
    return render_template(
        "results_live.html",
        live_report=data,
        coaching_tips=coaching_tips,
    )


def _live_worker(
    exercise: str,
    calibration_seconds: int,
    confidence_threshold: float,
    duration_seconds: int,
    camera_index: int,
) -> None:
    def frame_callback(rgb: np.ndarray) -> None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if ok:
            with _live_lock:
                _live_state["frame_jpeg"] = buf.tobytes()

    def stats_callback(el: float, dur: int, reps: int, low_conf: int, last_score: float | None) -> None:
        with _live_lock:
            _live_state["stats"] = {
                "elapsed": el,
                "duration_seconds": dur,
                "reps": reps,
                "low_confidence_frames": low_conf,
                "last_rep_score": last_score,
            }

    try:
        report = run_live_webcam_assessment(
            exercise,
            calibration_seconds,
            confidence_threshold,
            duration_seconds,
            camera_index,
            frame_callback=frame_callback,
            stats_callback=stats_callback,
            status_callback=None,
        )
        save_report_json(report, LIVE_SESSION_REPORT)
        with _live_lock:
            _live_state["report"] = report
    except Exception as exc:
        with _live_lock:
            _live_state["error"] = str(exc)
    finally:
        with _live_lock:
            _live_state["mode"] = "finished"


@app.post("/submit")
def submit():
    exercise = request.form.get("exercise") or "pushup"
    if exercise not in list_exercise_keys():
        exercise = "pushup"

    try:
        calibration_seconds = int(request.form.get("calibration_seconds", 3))
    except (TypeError, ValueError):
        calibration_seconds = 3
    calibration_seconds = max(3, min(15, calibration_seconds))

    try:
        confidence_threshold = float(request.form.get("confidence_threshold", 0.5))
    except (TypeError, ValueError):
        confidence_threshold = 0.5
    confidence_threshold = max(0.3, min(0.9, confidence_threshold))

    source_mode = request.form.get("source_mode") or MODE_LIBRARY
    action = request.form.get("action") or "run"

    if source_mode == MODE_LIVE and action == "live":
        try:
            duration = int(request.form.get("live_duration_seconds", 20))
        except (TypeError, ValueError):
            duration = 20
        duration = max(10, min(120, duration))
        try:
            cam = int(request.form.get("camera_index", 0))
        except (TypeError, ValueError):
            cam = 0
        cam = max(0, min(5, cam))

        with _live_lock:
            if _live_state["mode"] == "running":
                flash("A live session is already running.", "warn")
                return redirect(url_for("index"))
            _reset_live_state_for_new_session()
            _live_state["mode"] = "running"

        thread = threading.Thread(
            target=_live_worker,
            kwargs={
                "exercise": exercise,
                "calibration_seconds": calibration_seconds,
                "confidence_threshold": confidence_threshold,
                "duration_seconds": duration,
                "camera_index": cam,
            },
            daemon=True,
        )
        thread.start()
        return redirect(url_for("live_page"))

    if action == "demo":
        vids = list_input_videos()
        demo_path = get_demo_input_video(vids)
        if demo_path is None:
            flash("No demo video found in input folder.", "error")
            return redirect(url_for("index"))
        processing_input = demo_path
    elif source_mode == MODE_UPLOAD:
        f = request.files.get("video_file")
        if not f or not f.filename:
            flash("Please upload a video before running analysis.", "error")
            return redirect(url_for("index"))
        UPLOAD_INPUT.write_bytes(f.read())
        processing_input = UPLOAD_INPUT
    elif source_mode == MODE_LIBRARY:
        name = request.form.get("selected_video") or ""
        if not name or not _allowed_video_basename(name):
            flash("Select a valid input video.", "error")
            return redirect(url_for("index"))
        processing_input = INPUT_DIR / name
    else:
        flash("Invalid source mode.", "error")
        return redirect(url_for("index"))

    try:
        report = process_video(
            video_path=str(processing_input),
            output_path=str(CANONICAL_OUTPUT),
            report_json_path=str(CANONICAL_REPORT),
            debug=False,
            calibration_seconds=calibration_seconds,
            confidence_threshold=confidence_threshold,
            exercise=exercise,
        )
    except Exception as exc:
        flash(f"Analysis failed: {exc}", "error")
        return redirect(url_for("index"))

    summary = report.get("summary", {}) or {}
    coaching_tips = build_holistic_feedback(report.get("exercise", "-"), summary)
    preview_path, preview_err = get_web_preview_video(CANONICAL_OUTPUT)
    video_url = url_for("video_preview") if preview_path.exists() else None
    download_url = url_for("video_download") if preview_path.exists() else None

    flash("Analysis complete" if action != "demo" else "Demo quick run complete", "success")
    return render_template(
        "results_video.html",
        video_report=report,
        coaching_tips=coaching_tips,
        preview_err=preview_err,
        video_url=video_url,
        download_url=download_url,
    )


@app.get("/live/view")
def live_page():
    with _live_lock:
        mode = _live_state["mode"]
    if mode not in ("running", "finished"):
        flash("Start a live session from the home page.", "warn")
        return redirect(url_for("index"))
    return render_template("live.html")


@app.get("/live/stream")
def live_stream():
    boundary = b"--frame\r\n"

    def generate():
        while True:
            with _live_lock:
                mode = _live_state["mode"]
                jpg = _live_state["frame_jpeg"]
            if jpg:
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            if mode in ("finished", "idle"):
                break
            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/live/status")
def live_status():
    with _live_lock:
        mode = _live_state["mode"]
        stats = dict(_live_state["stats"])
        err = _live_state.get("error")
    return jsonify(
        {
            "mode": mode,
            "done": mode == "finished",
            "session_active": mode == "running",
            "stats": stats,
            "error": err,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
