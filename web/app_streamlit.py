import json
from pathlib import Path
import re
import sys
import subprocess
import time
from typing import Any

import cv2
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.exercise_profiles import list_exercise_keys
from src.exercise_profiles import get_exercise_profile, build_analyzer
from src.pose_estimator import PoseEstimator
from src.angle_engine import get_all_angles
from src.temporal_engine import TemporalEngine
from src.rep_counter import RepCounter
from src.scorer import score_rep, summarize_session
from src.exercise_rep_validation import ExerciseRepValidator
from video_processor import process_video
from video_processor import (
    get_primary_angle_with_confidence,
    _estimate_floor_y,
    _compute_pushup_floor_clearance,
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
WEB_OUTPUT = OUTPUT_DIR / "output_web.mp4"
CANONICAL_REPORT = REPORT_DIR / "report.json"
PICS_DIR = BASE_DIR / "pics"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

MODE_LIBRARY = "Input folder"
MODE_UPLOAD = "Upload file"
MODE_LIVE = "Live webcam"
FEEDBACK_TEXT_OCEAN_BLUE = (148, 105, 0)
FEEDBACK_BG_WHITE = (255, 255, 255)
FEEDBACK_BORDER_BLACK = (0, 0, 0)


st.set_page_config(page_title="FormAI Coach", page_icon="AI", layout="wide")


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


def inject_ui_theme() -> None:
    """Apply a clean, hackathon-grade visual style without adding UI clutter."""
    st.markdown(
        """
        <style>
        :root {
            --primary: #0b3a53;
            --accent: #17a2b8;
            --surface: #f7fafc;
            --text-muted: #52606d;
            --ok: #0f8b3d;
            --warn: #b76e00;
            --bad: #b42318;
        }
        .main .block-container {
            max-width: 1200px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(120deg, #f2f8fb, #ffffff);
            border: 1px solid #dde7ee;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            color: var(--primary);
            font-size: 1.9rem;
            letter-spacing: 0.2px;
        }
        .hero p {
            margin: 0.35rem 0 0 0;
            color: var(--text-muted);
            font-size: 0.97rem;
        }
        .step-card {
            background: var(--surface);
            border: 1px solid #e3edf3;
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.7rem;
        }
        .step-card h3 {
            margin: 0;
            font-size: 1rem;
            color: var(--primary);
        }
        .step-card p {
            margin: 0.2rem 0 0 0;
            color: var(--text-muted);
            font-size: 0.92rem;
        }
        .result-chip {
            display: inline-block;
            padding: 0.22rem 0.5rem;
            border: 1px solid #d7e3eb;
            border-radius: 999px;
            margin: 0.15rem 0.3rem 0 0;
            background: #f8fcff;
            color: #204055;
            font-size: 0.83rem;
        }
        .exec-card {
            background: linear-gradient(120deg, #f8fbfd, #ffffff);
            border: 1px solid #dbe8f0;
            border-radius: 14px;
            padding: 0.8rem 0.95rem;
            margin: 0.35rem 0 0.8rem 0;
        }
        .exec-title {
            margin: 0;
            color: var(--primary);
            font-size: 1rem;
            font-weight: 700;
        }
        .exec-line {
            margin: 0.25rem 0 0 0;
            color: #24485f;
            font-size: 0.93rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>FormAI Exercise Coach</h1>
          <p>Upload or stream your workout, get rep counts, quality scores, and coaching feedback in one clean flow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_metrics(exercise_key: str, summary: dict) -> None:
    total_reps = int(summary.get("total_reps", 0) or 0)
    valid_reps = int(summary.get("valid_reps", 0) or 0)
    avg_score = float(summary.get("avg_score", 0) or 0)
    consistency = (valid_reps / total_reps * 100.0) if total_reps else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Exercise", exercise_key or "-")
    m2.metric("Total Reps", total_reps)
    m3.metric("Valid Reps", valid_reps)
    m4.metric("Avg Score", int(round(avg_score)))

    st.caption(f"Rep consistency: {consistency:.0f}%")


def render_executive_summary(exercise_key: str, summary: dict) -> None:
    total_reps = int(summary.get("total_reps", 0) or 0)
    valid_reps = int(summary.get("valid_reps", 0) or 0)
    avg_score = float(summary.get("avg_score", 0) or 0)
    consistency = (valid_reps / total_reps * 100.0) if total_reps else 0.0

    if total_reps == 0:
        verdict = "No measurable reps detected yet"
    elif avg_score >= 85 and consistency >= 85:
        verdict = "Excellent form session"
    elif avg_score >= 70 and consistency >= 70:
        verdict = "Solid form with room to improve"
    else:
        verdict = "Needs focused technique work"

    st.markdown(
        (
            '<div class="exec-card">'
            '<p class="exec-title">Executive Summary</p>'
            f'<p class="exec-line">{exercise_key} • {verdict}</p>'
            f'<p class="exec-line">Reps: {total_reps} | Valid: {valid_reps} | Consistency: {consistency:.0f}% | Avg score: {int(round(avg_score))}</p>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def render_issue_chips(summary: dict) -> None:
    top_issues = summary.get("top_issues", []) or []
    st.markdown("### Top Issues")
    if not top_issues:
        st.write("No major issues detected.")
        return

    chips = "".join([f'<span class="result-chip">{issue}</span>' for issue in top_issues])
    st.markdown(chips, unsafe_allow_html=True)


def render_coaching_feedback(exercise_key: str, summary: dict) -> None:
    st.markdown("### Coaching Feedback")
    for tip in build_holistic_feedback(exercise_key, summary):
        st.write(f"- {tip}")


def render_video_report(report: dict, preview_path: Path | None = None, preview_err: str | None = None) -> None:
    summary = report.get("summary", {}) or {}
    exercise_key = report.get("exercise", "-")

    render_executive_summary(exercise_key, summary)

    if preview_path is not None:
        if preview_err:
            st.warning(preview_err)
        render_video(preview_path, "Analyzed output")

    render_summary_metrics(exercise_key, summary)
    render_issue_chips(summary)
    render_coaching_feedback(exercise_key, summary)

    confidence = report.get("confidence", {}) or {}
    with st.expander("Technical diagnostics", expanded=False):
        st.write(f"Readiness rejected frames: {confidence.get('readiness_rejected_frames', 0)}")
        st.write(f"Floor-clearance rejected frames: {confidence.get('floor_clearance_rejected_frames', 0)}")
        posture_counts = confidence.get("posture_hint_counts", {})
        if posture_counts:
            st.write("Posture hint counts:")
            st.write(posture_counts)


def render_live_report(report: dict) -> None:
    summary = report.get("summary", {}) or {}
    exercise_key = report.get("exercise", "-")
    render_executive_summary(exercise_key, summary)
    render_summary_metrics(exercise_key, summary)
    render_coaching_feedback(exercise_key, summary)

    with st.expander("Live session diagnostics", expanded=False):
        st.write(report.get("live", {}))


def build_holistic_feedback(exercise_key: str, summary: dict) -> list[str]:
    """Create session-level actionable feedback from summary metrics."""
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


inject_ui_theme()
render_hero()


def _clean_feedback_for_overlay(message: str | None, max_chars: int = 72) -> str:
    """Normalize feedback text so webcam overlays stay readable."""
    if message is None:
        return ""

    text = str(message).strip()
    if not text:
        return ""

    # OpenCV Hershey fonts cannot render Unicode icons (✓ ⚠ ❌ 🚨), so keep ASCII only.
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


def _draw_feedback_badge(frame, text: str, x: int, y: int, font_scale: float = 0.54, thickness: int = 1) -> None:
    """Draw compact feedback badge with ocean-blue text and subtle white background."""
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
):
    """Run timed live webcam assessment and return report dict."""
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

    frame_box = st.empty()
    stats_box = st.empty()
    status_box = st.empty()
    status_box.info("Live assessment running. Keep full body visible from side angle.")

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
                primary_angle, primary_confidence, _ = get_primary_angle_with_confidence(profile, landmarks, angles)

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
                    cv2.putText(frame, validity_text, (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.62, validity_color, 2)

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
            frame_box.image(rgb_frame, channels="RGB", use_container_width=True)

            stats_box.markdown(
                f"**Elapsed:** {elapsed:.1f}s / {duration_seconds}s | "
                f"**Detected reps:** {reps} | "
                f"**Low-confidence frames:** {low_confidence_frames} | "
                f"**Last rep score:** {('--' if last_rep_score is None else int(round(last_rep_score)))}"
            )
    finally:
        cap.release()
        status_box.success("Live session finished.")

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


def render_video(path: Path, label: str):
    """Render video robustly and provide a download fallback."""
    if not path.exists():
        st.warning(f"{label} not found at {path.relative_to(BASE_DIR)}")
        return

    video_bytes = path.read_bytes()
    if not video_bytes:
        st.warning(f"{label} file is empty: {path.relative_to(BASE_DIR)}")
        return

    st.video(video_bytes, format="video/mp4")
    st.caption(f"{label}: {path.relative_to(BASE_DIR)} ({len(video_bytes) / (1024*1024):.2f} MB)")
    st.download_button(
        label=f"Download {label}",
        data=video_bytes,
        file_name=path.name,
        mime="video/mp4",
        use_container_width=True,
    )


def list_input_videos() -> list[Path]:
    """List videos available in input directory."""
    videos = [p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos, key=lambda p: p.name.lower())


def get_demo_input_video(available_videos: list[Path]) -> Path | None:
    """Pick a best-effort demo video from input folder for one-click quick run."""
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
    """Find best matching guide image in pics folder for selected exercise."""
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
        "bicep curl": {"bicep curl", "biceps curl", "bicep curls", "biceps curls", "curl", "curls"},
        "shoulder press": {"shoulder press", "shoulder presses", "overhead press"},
        "situp": {"situp", "sit up", "situps", "sit ups", "sit-up", "sit-ups"},
        "mountain climber": {"mountain climber", "mountain climbers", "climber", "climbers"},
    }

    target_aliases = set()
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


def get_web_preview_video(source_path: Path) -> tuple[Path, str | None]:
    """Return browser-friendly video path, transcoding if needed."""
    if not source_path.exists():
        return source_path, "Output video file not found"

    if imageio_ffmpeg is None:
        return source_path, "Preview transcoding unavailable: install imageio-ffmpeg for browser-optimized output"

    # Reuse converted file only when it is strictly newer than source.
    # This avoids stale preview reuse when both files share same-second mtime.
    if (
        WEB_OUTPUT.exists()
        and WEB_OUTPUT.stat().st_size > 0
        and WEB_OUTPUT.stat().st_mtime > source_path.stat().st_mtime
    ):
        return WEB_OUTPUT, None

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(source_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(WEB_OUTPUT),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if WEB_OUTPUT.exists() and WEB_OUTPUT.stat().st_size > 0:
            return WEB_OUTPUT, None
        return source_path, "Converted preview video was empty"
    except Exception as exc:
        return source_path, f"Could not transcode for web preview: {exc}"

with st.sidebar:
    st.header("Session Setup")
    exercise = st.selectbox("Exercise", options=list_exercise_keys(), index=list_exercise_keys().index("pushup"))
    with st.expander("Advanced detection settings", expanded=False):
        calibration_seconds = st.slider(
            "Calibration seconds",
            min_value=3,
            max_value=15,
            value=3,
            help="How long FormAI watches your early movement to adapt thresholds. Longer = more stable, slower startup.",
        )
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.30,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="How certain pose landmarks must be before counting/analyzing a frame. Higher = safer but can skip more frames.",
        )

    guide_image_path = get_guide_image_for_exercise(exercise)
    if guide_image_path is not None:
        st.markdown("### Guide")
        st.image(str(guide_image_path), caption=f"Reference: {guide_image_path.name}", use_container_width=True)
    else:
        st.info("No guide image found in pics folder for this exercise yet.")

source_mode = st.radio(
    "How do you want to run analysis?",
    options=[MODE_LIBRARY, MODE_UPLOAD, MODE_LIVE],
    horizontal=True,
)
uploaded_file = None
available_videos = list_input_videos()
demo_input_path = get_demo_input_video(available_videos)
selected_input_name = None
live_duration_seconds = 20
camera_index = 0

if source_mode == MODE_LIVE:
    live_duration_seconds = st.slider("Live session duration (seconds)", min_value=10, max_value=120, value=20, step=5)
    camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)

if source_mode == MODE_LIBRARY:
    if available_videos:
        names = [p.name for p in available_videos]
        default_name = "input.mp4" if "input.mp4" in names else names[0]
        selected_input_name = st.selectbox("Choose existing input video", options=names, index=names.index(default_name))
    else:
        st.warning("No videos found in input folder. Upload a video to continue.")

if source_mode == MODE_UPLOAD:
    uploaded_file = st.file_uploader("Upload MP4/MOV video", type=["mp4", "mov", "mkv", "avi"])

left, right = st.columns([1.0, 1.35])

with left:
    st.markdown('<div class="step-card"><h3>Step 1: Input</h3><p>Pick a source, verify preview, then run analysis.</p></div>', unsafe_allow_html=True)

    input_path = CANONICAL_INPUT

    if source_mode == MODE_LIBRARY:
        if selected_input_name is not None:
            input_path = INPUT_DIR / selected_input_name
            st.success(f"Using {input_path.relative_to(BASE_DIR)}")
            render_video(input_path, "Input preview")
        else:
            st.error("No input video selected. Upload a video instead.")
    elif source_mode == MODE_UPLOAD:
        if uploaded_file is not None:
            UPLOAD_INPUT.write_bytes(uploaded_file.read())
            input_path = UPLOAD_INPUT
            st.success(f"Uploaded to {UPLOAD_INPUT.relative_to(BASE_DIR)}")
            render_video(UPLOAD_INPUT, "Uploaded input")
        else:
            st.info("Upload a video to continue.")
    else:
        st.info("Webcam mode uses your live camera feed.")
        st.caption("Tip: place camera side-on with full body visible for best push-up counting.")

    run_clicked = st.button(
        "Start Live Assessment" if source_mode == MODE_LIVE else "Run Analysis",
        type="primary",
        use_container_width=True,
    )

    demo_clicked = False
    if source_mode != MODE_LIVE:
        demo_label = "Demo Day Quick Run"
        if demo_input_path is not None:
            st.caption(f"One-click demo uses: {demo_input_path.name}")
            demo_clicked = st.button(demo_label, use_container_width=True)
        else:
            st.caption("Add a sample video in input/ to enable Demo Day Quick Run.")

with right:
    st.markdown('<div class="step-card"><h3>Step 2: Output</h3><p>Review metrics, top issues, and actionable feedback.</p></div>', unsafe_allow_html=True)

    if run_clicked or demo_clicked:
        if source_mode == MODE_LIVE:
            with st.spinner("Starting webcam assessment..."):
                report = run_live_webcam_assessment(
                    exercise=exercise,
                    calibration_seconds=calibration_seconds,
                    confidence_threshold=confidence_threshold,
                    duration_seconds=live_duration_seconds,
                    camera_index=int(camera_index),
                )

            st.success("Live assessment complete")
            render_live_report(report)

        elif source_mode == MODE_UPLOAD and uploaded_file is None:
            st.error("Please upload a video before running analysis.")
        elif demo_clicked and demo_input_path is None:
            st.error("No demo video found in input folder.")
        elif source_mode == MODE_LIBRARY and not input_path.exists():
            st.error("Selected input video does not exist.")
        else:
            processing_input = demo_input_path if demo_clicked and demo_input_path is not None else input_path
            with st.spinner("Analyzing video... this may take a minute."):
                report = process_video(
                    video_path=str(processing_input),
                    output_path=str(CANONICAL_OUTPUT),
                    report_json_path=str(CANONICAL_REPORT),
                    debug=False,
                    calibration_seconds=calibration_seconds,
                    confidence_threshold=confidence_threshold,
                    exercise=exercise,
                )

            st.success("Demo quick run complete" if demo_clicked else "Analysis complete")
            preview_path, preview_err = get_web_preview_video(CANONICAL_OUTPUT)
            render_video_report(report, preview_path=preview_path, preview_err=preview_err)

    elif CANONICAL_OUTPUT.exists() and CANONICAL_REPORT.exists():
        st.info("Showing latest saved output.")
        preview_path, preview_err = get_web_preview_video(CANONICAL_OUTPUT)
        data = json.loads(CANONICAL_REPORT.read_text(encoding="utf-8"))
        render_video_report(data, preview_path=preview_path, preview_err=preview_err)
    else:
        st.info("Run analysis to see a professional session report here.")

st.markdown("---")
st.caption("Outputs are saved to output/output.mp4 and report/report.json")
