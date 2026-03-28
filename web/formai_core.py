"""Shared FormAI UI logic (no Streamlit / Flask)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.angle_engine import get_all_angles
from src.exercise_profiles import build_analyzer, get_exercise_profile
from src.exercise_rep_validation import ExerciseRepValidator
from src.pose_estimator import PoseEstimator
from src.rep_counter import RepCounter
from src.scorer import score_rep, summarize_session
from src.temporal_engine import TemporalEngine
from video_processor import (
    _compute_pushup_floor_clearance,
    _estimate_floor_y,
    get_primary_angle_with_confidence,
    is_pushup_ready_for_count,
)

try:
    import imageio_ffmpeg
except ModuleNotFoundError:
    imageio_ffmpeg = None

INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
REPORT_DIR = BASE_DIR / "report"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_INPUT = INPUT_DIR / "input.mp4"
UPLOAD_INPUT = INPUT_DIR / "uploaded.mp4"
CANONICAL_OUTPUT = OUTPUT_DIR / "output.mp4"
_TRANSCODE_LOCK = threading.Lock()
CANONICAL_REPORT = REPORT_DIR / "report.json"
LIVE_SESSION_REPORT = REPORT_DIR / "live_session.json"
PICS_DIR = BASE_DIR / "pics"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

MODE_LIBRARY = "Input folder"
MODE_UPLOAD = "Upload file"
MODE_LIVE = "Live webcam"

FEEDBACK_TEXT_OCEAN_BLUE = (148, 105, 0)
FEEDBACK_BG_WHITE = (255, 255, 255)
FEEDBACK_BORDER_BLACK = (0, 0, 0)

ISSUE_TO_TIP = {
    "shallow_depth": "Increase range of motion and hit deeper reps consistently.",
    "short_range_of_motion": "Complete full contraction and full extension for each rep.",
    "no_lockout": "Finish each rep fully at the top before starting the next one.",
    "hip_sag": "Keep your core braced and maintain a more stable torso.",
    "forward_lean": "Keep your torso more upright to reduce compensation.",
    "inconsistent_depth": "Keep tempo steady and match depth across reps.",
    "momentum_cheat": "Slow down slightly and avoid using momentum to move weight.",
    "neck_strain": "Keep neck neutral and avoid pulling from the neck.",
    "knee_valgus": "Track knees in line with toes to improve joint alignment.",
    "core_instability": "Brace your core before each rep and control transitions.",
    "joint_stress": "Avoid forcing end range; use smooth, controlled depth.",
}


def build_holistic_feedback(exercise_key: str, summary: dict) -> list[str]:
    tips = []

    for tip in summary.get("coaching_tips", []) or []:
        if tip and tip not in tips:
            tips.append(str(tip))

    raw_issue_counts = summary.get("issue_counts", {}) or {}
    issue_counts: dict[str, int] = {}
    if isinstance(raw_issue_counts, dict):
        for key, value in raw_issue_counts.items():
            if isinstance(key, str):
                try:
                    issue_counts[key] = int(value)
                except (TypeError, ValueError):
                    continue

    if issue_counts:
        top_issue_keys = sorted(issue_counts.keys(), key=lambda k: issue_counts[k], reverse=True)[:3]
        for key in top_issue_keys:
            mapped_tip = ISSUE_TO_TIP.get(key)
            if mapped_tip and mapped_tip not in tips:
                tips.append(mapped_tip)

    total_reps = int(summary.get("total_reps", 0) or 0)
    valid_reps = int(summary.get("valid_reps", 0) or 0)
    if total_reps > 0:
        consistency = (valid_reps / total_reps) * 100.0
        if consistency < 70:
            tips.append("Focus on consistent setup each rep before increasing speed.")
        elif consistency < 90:
            tips.append("Good session overall; prioritize consistency on every rep.")

    avg_score = float(summary.get("avg_score", 0) or 0)
    if avg_score >= 90 and total_reps > 0:
        tips.append("Strong form overall. Next step: maintain quality at a steady tempo.")

    if not tips:
        tips.append("No major issues detected. Keep this setup and range for future sets.")

    return tips[:5]


def list_input_videos() -> list[Path]:
    videos = [p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos, key=lambda p: p.name.lower())


def get_demo_input_video(available_videos: list[Path]) -> Path | None:
    preferred_names = ["input.mp4", "uploaded.mp4"]
    by_name = {p.name.lower(): p for p in available_videos}
    for name in preferred_names:
        if name in by_name:
            return by_name[name]
    return available_videos[0] if available_videos else None


def _normalize_exercise_name(value: str) -> str:
    cleaned = value.lower().replace("_", " ").replace("-", " ").strip()
    cleaned = " ".join(cleaned.split())
    return cleaned


def get_guide_image_for_exercise(exercise_key: str) -> Path | None:
    if not PICS_DIR.exists():
        return None

    files = [p for p in PICS_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        return None

    target = _normalize_exercise_name(exercise_key)

    synonyms = {
        "pushup": {"pushup", "push up", "pushups", "push ups"},
        "squat": {"squat", "squats"},
        "lunge": {"lunge", "lunges"},
        "bicep curl": {
            "bicep curl",
            "biceps curl",
            "bicep curls",
            "biceps curls",
            "curl",
            "curls",
        },
        "shoulder press": {"shoulder press", "shoulder presses", "overhead press"},
        "situp": {"situp", "sit up", "situps", "sit ups", "sit-up", "sit-ups"},
        "mountain climber": {"mountain climber", "mountain climbers", "climber", "climbers"},
    }

    if target in synonyms:
        target_aliases = synonyms[target]
    else:
        target_aliases = {target}

    best_match = None
    for file_path in files:
        name = _normalize_exercise_name(file_path.stem)
        if name in target_aliases:
            return file_path
        if any(alias in name for alias in target_aliases):
            best_match = file_path

    return best_match


def _prune_stale_web_previews(keep: Path) -> None:
    """Remove older output_web_*.mp4 caches so the folder does not grow forever."""
    for p in OUTPUT_DIR.glob("output_web_*.writing.mp4"):
        try:
            p.unlink()
        except OSError:
            pass
    for p in OUTPUT_DIR.glob("output_web_*.mp4"):
        if p.name.endswith(".writing.mp4"):
            continue
        try:
            if p.resolve() != keep.resolve():
                p.unlink()
        except OSError:
            pass


def get_web_preview_video(source_path: Path) -> tuple[Path, str | None]:
    """
    Produce an H.264 / yuv420p MP4 for HTML5 playback.

    Cache filename is derived from source mtime + size so we never overwrite a file
    the browser may still have open on Windows (avoids FFmpeg/transcoder errors).
    """
    if not source_path.exists():
        return source_path, "Output video file not found"

    if imageio_ffmpeg is None:
        return source_path, "Preview transcoding unavailable: install imageio-ffmpeg for browser-optimized output"

    st = source_path.stat()
    cache_path = OUTPUT_DIR / f"output_web_{int(st.st_mtime)}_{st.st_size}.mp4"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path, None

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    # Filename must end in .mp4 so FFmpeg chooses the MP4 muxer (.mp4.part breaks format detection).
    tmp_out = cache_path.parent / f"{cache_path.stem}.writing{cache_path.suffix}"
    vcodec_attempts: list[list[str]] = [
        ["-c:v", "libx264", "-crf", "23", "-preset", "veryfast"],
    ]
    if sys.platform == "win32":
        vcodec_attempts.append(["-c:v", "h264_mf"])

    with _TRANSCODE_LOCK:
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return cache_path, None

        try:
            if tmp_out.exists():
                tmp_out.unlink()
        except OSError:
            pass

        last_stderr = ""
        for vopts in vcodec_attempts:
            cmd = [
                ffmpeg_exe,
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-i",
                str(source_path),
                *vopts,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                "-f",
                "mp4",
                str(tmp_out),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            last_stderr = (result.stderr or result.stdout or "").strip()
            if result.returncode == 0 and tmp_out.exists() and tmp_out.stat().st_size > 0:
                try:
                    os.replace(tmp_out, cache_path)
                except OSError:
                    try:
                        if cache_path.exists():
                            cache_path.unlink()
                        os.replace(tmp_out, cache_path)
                    except OSError as exc:
                        return source_path, (
                            "Could not finalize web preview file (is it open in another app?): "
                            f"{exc}"
                        )
                _prune_stale_web_previews(cache_path)
                return cache_path, None

        detail = last_stderr[:1800] if last_stderr else "unknown error (no stderr)"
        try:
            if tmp_out.exists():
                tmp_out.unlink()
        except OSError:
            pass
        return source_path, (
            "Could not transcode for web preview (tried libx264"
            + (" and h264_mf" if sys.platform == "win32" else "")
            + f"). FFmpeg said:\n{detail}"
        )


def _clean_feedback_for_overlay(message: str | None, max_chars: int = 72) -> str:
    if message is None:
        return ""

    text = str(message).strip()
    if not text:
        return ""

    text = text.encode("ascii", "ignore").decode("ascii")

    text = re.sub(r"^[^A-Za-z0-9]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _feedback_lines_for_overlay(feedback: list[str], max_lines: int = 2, max_chars: int = 66) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for message in feedback or []:
        line = _clean_feedback_for_overlay(message, max_chars=max_chars)
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
        if len(lines) >= max_lines:
            break
    return lines


def _draw_feedback_badge(
    frame, text: str, x: int, y: int, font_scale: float = 0.54, thickness: int = 1
) -> None:
    if not text:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x = 4
    pad_y = 3

    top_left = (x - pad_x, y - text_h - pad_y)
    bottom_right = (x + text_w + pad_x, y + baseline + pad_y)

    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, FEEDBACK_BG_WHITE, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, top_left, bottom_right, FEEDBACK_BORDER_BLACK, 1)
    cv2.putText(frame, text, (x, y), font, font_scale, FEEDBACK_TEXT_OCEAN_BLUE, thickness)


def run_live_webcam_assessment(
    exercise: str,
    calibration_seconds: int,
    confidence_threshold: float,
    duration_seconds: int,
    camera_index: int,
    *,
    frame_callback: Callable[[Any], None] | None = None,
    stats_callback: Callable[..., None] | None = None,
    status_callback: Callable[[str, str], None] | None = None,
) -> dict:
    """
    Run timed live webcam assessment. Optional callbacks:
    - frame_callback(rgb_ndarray)
    - stats_callback(elapsed, duration_seconds, reps, low_confidence_frames, last_rep_score)
    - status_callback(message, level)  level: info | success | error
    """
    profile = get_exercise_profile(exercise)
    estimator = PoseEstimator()
    temporal = TemporalEngine(buffer_size=2 if profile.key == "pushup" else 3)
    counter = RepCounter(
        descent_trigger=profile.default_descent_trigger,
        ascent_threshold=profile.default_ascent_threshold,
        require_initial_lockout=(profile.key == "pushup"),
        initial_lockout_frames=2,
    )
    analyzer: Any = build_analyzer(profile.key)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise ValueError("Could not open webcam. Check camera permission and index.")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    reps = 0
    rep_reports = []
    last_feedback = []
    last_rep_score = None
    last_rep_valid = None
    last_validation_reason = ""
    floor_y_estimate = None
    frame_count = 0
    low_confidence_frames = 0
    calibration_frames = max(10, int(calibration_seconds * fps))
    calibration_angles = []

    if status_callback:
        status_callback(
            "Live assessment running. Keep full body visible from side angle.",
            "info",
        )

    start_time = time.time()
    elapsed = 0.0
    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= float(duration_seconds):
                break

            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1

            results = estimator.process_frame(frame)
            landmarks = estimator.get_landmarks(results, frame.shape)

            if landmarks is not None:
                angles = get_all_angles(landmarks)
                primary_angle, primary_confidence, _ = get_primary_angle_with_confidence(
                    profile, landmarks, angles
                )

                if primary_angle is not None and frame_count <= calibration_frames:
                    calibration_angles.append(primary_angle)

                analysis_conf_ok = primary_angle is not None and primary_confidence >= confidence_threshold
                count_conf_threshold = confidence_threshold
                if profile.key == "pushup":
                    count_conf_threshold = max(0.40, confidence_threshold - 0.10)
                count_conf_ok = primary_angle is not None and primary_confidence >= count_conf_threshold

                if not count_conf_ok:
                    low_confidence_frames += 1

                smooth_angle = temporal.smooth(primary_angle) if count_conf_ok else temporal.last_valid
                stage = temporal.detect_stage(smooth_angle)

                if analysis_conf_ok:
                    analyzer.collect(landmarks, angles)

                if profile.key == "pushup":
                    floor_y_estimate = _estimate_floor_y(landmarks, floor_y_estimate)
                    floor_clearance = _compute_pushup_floor_clearance(landmarks, floor_y_estimate)
                    _ = is_pushup_ready_for_count(angles, floor_clearance=floor_clearance)

                if count_conf_ok and counter.update(smooth_angle):
                    if profile.key == "pushup" and reps == 0 and frame_count <= max(30, int(1.0 * fps)):
                        analyzer.reset()
                        continue

                    feedback = analyzer.evaluate()
                    rep_score = score_rep(feedback)
                    should_count_rep, validation_reason = ExerciseRepValidator.validate_rep(
                        profile.key, rep_score, feedback
                    )

                    reps += 1
                    if not should_count_rep:
                        rep_score["is_valid"] = False
                        invalid_reasons = set(rep_score.get("invalid_reasons", []))
                        invalid_reasons.add("rep_rejected")
                        rep_score["invalid_reasons"] = sorted(invalid_reasons)

                    rep_reports.append(
                        {
                            "rep_number": reps,
                            "feedback": feedback,
                            "counted_valid": bool(should_count_rep),
                            "validation_reason": validation_reason,
                            **rep_score,
                        }
                    )
                    last_feedback = feedback[:2]
                    last_rep_score = rep_score.get("score")
                    last_rep_valid = bool(should_count_rep)
                    last_validation_reason = validation_reason
                    analyzer.reset()

                frame = estimator.draw_skeleton(frame, results)
                cv2.putText(frame, f"Reps: {reps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Stage: {stage}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                score_text = "Score: --"
                score_color = (220, 220, 220)
                if last_rep_score is not None:
                    score_text = f"Score: {int(round(last_rep_score))}/100"
                    if last_rep_score >= 80:
                        score_color = (0, 200, 0)
                    elif last_rep_score >= 60:
                        score_color = (0, 180, 255)
                    else:
                        score_color = (0, 0, 255)
                cv2.putText(frame, score_text, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.72, score_color, 2)

                if last_rep_valid is not None:
                    validity_text = "Quality: Valid rep" if last_rep_valid else "Quality: Needs correction"
                    validity_color = (0, 200, 0) if last_rep_valid else (0, 0, 255)
                    cv2.putText(
                        frame, validity_text, (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.62, validity_color, 2
                    )

                overlay_feedback = _feedback_lines_for_overlay(last_feedback, max_lines=2, max_chars=56)
                if (not last_rep_valid) and last_validation_reason:
                    reason_line = _clean_feedback_for_overlay(last_validation_reason, max_chars=56)
                    if reason_line and reason_line.lower() not in {line.lower() for line in overlay_feedback}:
                        overlay_feedback.insert(0, reason_line)
                        overlay_feedback = overlay_feedback[:2]

                if overlay_feedback:
                    _draw_feedback_badge(frame, overlay_feedback[0], 20, 175, font_scale=0.54, thickness=1)
                if len(overlay_feedback) > 1:
                    _draw_feedback_badge(frame, overlay_feedback[1], 20, 201, font_scale=0.54, thickness=1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_callback is not None:
                frame_callback(rgb_frame)

            if stats_callback is not None:
                stats_callback(
                    elapsed,
                    duration_seconds,
                    reps,
                    low_confidence_frames,
                    last_rep_score,
                )
    finally:
        cap.release()
        if status_callback:
            status_callback("Live session finished.", "success")

    summary = summarize_session(rep_reports)
    summary["total_reps"] = reps

    return {
        "exercise": profile.key,
        "summary": summary,
        "rep_reports": rep_reports,
        "live": {
            "duration_seconds": duration_seconds,
            "fps_assumed": fps,
            "processed_frames": frame_count,
            "low_confidence_frames": low_confidence_frames,
            "calibration_samples": len(calibration_angles),
        },
    }


def save_report_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_report_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
