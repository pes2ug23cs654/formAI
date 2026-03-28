import json
from pathlib import Path
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


st.set_page_config(page_title="FormAI Coach", page_icon="AI", layout="wide")
st.title("FormAI Exercise Coach")
st.caption("Upload a workout video, choose exercise type, and get analyzed output with coaching feedback.")


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
                    analyzer.reset()

                frame = estimator.draw_skeleton(frame, results)
                cv2.putText(frame, f"Reps: {reps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Stage: {stage}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if last_feedback:
                    cv2.putText(frame, str(last_feedback[0])[:70], (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_box.image(rgb_frame, channels="RGB", use_container_width=True)

            stats_box.markdown(
                f"**Elapsed:** {elapsed:.1f}s / {duration_seconds}s | "
                f"**Detected reps:** {reps} | "
                f"**Low-confidence frames:** {low_confidence_frames}"
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
    st.header("Run Settings")
    exercise = st.selectbox("Exercise", options=list_exercise_keys(), index=list_exercise_keys().index("pushup"))
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
        st.markdown("### Exercise Guide")
        st.image(str(guide_image_path), caption=f"Reference: {guide_image_path.name}", use_container_width=True)
    else:
        st.info("No guide image found in pics folder for this exercise yet.")

source_mode = st.radio(
    "Input Source",
    options=["Use video from input folder", "Upload new video", "Live webcam assessment"],
    horizontal=True,
)
uploaded_file = None
available_videos = list_input_videos()
selected_input_name = None
live_duration_seconds = 20
camera_index = 0

if source_mode == "Live webcam assessment":
    live_duration_seconds = st.slider("Live session duration (seconds)", min_value=10, max_value=120, value=20, step=5)
    camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)

if source_mode == "Use video from input folder":
    if available_videos:
        names = [p.name for p in available_videos]
        default_name = "input.mp4" if "input.mp4" in names else names[0]
        selected_input_name = st.selectbox("Choose existing input video", options=names, index=names.index(default_name))
    else:
        st.warning("No videos found in input folder. Upload a video to continue.")

if source_mode == "Upload new video":
    uploaded_file = st.file_uploader("Upload MP4/MOV video", type=["mp4", "mov", "mkv", "avi"])

left, right = st.columns([1, 1])

with left:
    st.subheader("Input")
    if guide_image_path is not None:
        st.markdown("### Form Guide")
        st.image(str(guide_image_path), use_container_width=True)

    input_path = CANONICAL_INPUT

    if source_mode == "Use video from input folder":
        if selected_input_name is not None:
            input_path = INPUT_DIR / selected_input_name
            st.success(f"Using {input_path.relative_to(BASE_DIR)}")
            render_video(input_path, "Input preview")
        else:
            st.error("No input video selected. Upload a video instead.")
    elif source_mode == "Upload new video":
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
        "Start Live Assessment" if source_mode == "Live webcam assessment" else "Run Analysis",
        type="primary",
        use_container_width=True,
    )

with right:
    st.subheader("Output")

    if run_clicked:
        if source_mode == "Live webcam assessment":
            with st.spinner("Starting webcam assessment..."):
                report = run_live_webcam_assessment(
                    exercise=exercise,
                    calibration_seconds=calibration_seconds,
                    confidence_threshold=confidence_threshold,
                    duration_seconds=live_duration_seconds,
                    camera_index=int(camera_index),
                )

            st.success("Live assessment complete")

            summary = report.get("summary", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Exercise", report.get("exercise", "-"))
            m2.metric("Total Reps", summary.get("total_reps", 0))
            m3.metric("Valid Reps", summary.get("valid_reps", 0))
            m4.metric("Avg Score", summary.get("avg_score", 0))

            live_info = report.get("live", {})
            with st.expander("Live Session Diagnostics", expanded=False):
                st.write(live_info)

            st.markdown("### Session Feedback")
            for tip in build_holistic_feedback(report.get("exercise", ""), summary):
                st.write(f"- {tip}")

        elif source_mode == "Upload new video" and uploaded_file is None:
            st.error("Please upload a video before running analysis.")
        elif source_mode == "Use video from input folder" and not input_path.exists():
            st.error("Selected input video does not exist.")
        else:
            with st.spinner("Analyzing video... this may take a minute."):
                report = process_video(
                    video_path=str(input_path),
                    output_path=str(CANONICAL_OUTPUT),
                    report_json_path=str(CANONICAL_REPORT),
                    debug=False,
                    calibration_seconds=calibration_seconds,
                    confidence_threshold=confidence_threshold,
                    exercise=exercise,
                )

            st.success("Analysis complete")
            preview_path, preview_err = get_web_preview_video(CANONICAL_OUTPUT)
            if preview_err:
                st.warning(preview_err)
            render_video(preview_path, "Analyzed output")

            summary = report.get("summary", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Exercise", report.get("exercise", "-"))
            m2.metric("Total Reps", summary.get("total_reps", 0))
            m3.metric("Valid Reps", summary.get("valid_reps", 0))
            m4.metric("Avg Score", summary.get("avg_score", 0))

            st.markdown("### Top Issues")
            top_issues = summary.get("top_issues", [])
            if top_issues:
                st.write(", ".join(top_issues))
            else:
                st.write("No major issues detected.")

            st.markdown("### Session Feedback")
            for tip in build_holistic_feedback(report.get("exercise", ""), summary):
                st.write(f"- {tip}")

            confidence = report.get("confidence", {})
            with st.expander("Readiness Diagnostics", expanded=False):
                st.write(f"Readiness rejected frames: {confidence.get('readiness_rejected_frames', 0)}")
                st.write(f"Floor-clearance rejected frames: {confidence.get('floor_clearance_rejected_frames', 0)}")
                posture_counts = confidence.get("posture_hint_counts", {})
                if posture_counts:
                    st.write("Posture hint counts:")
                    st.write(posture_counts)

            # Per-rep feedback intentionally hidden in demo UX; summary guidance is shown above.

    elif CANONICAL_OUTPUT.exists() and CANONICAL_REPORT.exists():
        st.info("Showing latest saved output.")
        preview_path, preview_err = get_web_preview_video(CANONICAL_OUTPUT)
        if preview_err:
            st.warning(preview_err)
        render_video(preview_path, "Latest output")
        data = json.loads(CANONICAL_REPORT.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Exercise", data.get("exercise", "-"))
        m2.metric("Total Reps", summary.get("total_reps", 0))
        m3.metric("Valid Reps", summary.get("valid_reps", 0))
        m4.metric("Avg Score", summary.get("avg_score", 0))

        coaching_tips = summary.get("coaching_tips", [])
        st.markdown("### Session Feedback")
        for tip in build_holistic_feedback(data.get("exercise", ""), summary):
            st.write(f"- {tip}")
    else:
        st.info("Run analysis to see results.")

st.markdown("---")
st.caption("Outputs are saved to output/output.mp4 and report/report.json")
