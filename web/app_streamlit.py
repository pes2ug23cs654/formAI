import json
import sys
from pathlib import Path

import streamlit as st

from formai_core import (
    BASE_DIR,
    CANONICAL_INPUT,
    CANONICAL_OUTPUT,
    CANONICAL_REPORT,
    INPUT_DIR,
    MODE_LIBRARY,
    MODE_LIVE,
    MODE_UPLOAD,
    UPLOAD_INPUT,
    build_holistic_feedback,
    get_demo_input_video,
    get_guide_image_for_exercise,
    get_web_preview_video,
    list_input_videos,
    run_live_webcam_assessment,
)
from src.exercise_profiles import list_exercise_keys
from video_processor import process_video

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

st.set_page_config(page_title="FormAI Coach", page_icon="AI", layout="wide")


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
            "</div>"
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


inject_ui_theme()
render_hero()


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


def run_streamlit_live(
    exercise: str,
    calibration_seconds: int,
    confidence_threshold: float,
    duration_seconds: int,
    camera_index: int,
):
    frame_box = st.empty()
    stats_box = st.empty()
    status_box = st.empty()

    def status_callback(message: str, level: str) -> None:
        if level == "info":
            status_box.info(message)
        elif level == "success":
            status_box.success(message)
        else:
            status_box.warning(message)

    return run_live_webcam_assessment(
        exercise,
        calibration_seconds,
        confidence_threshold,
        duration_seconds,
        camera_index,
        frame_callback=lambda rgb: frame_box.image(rgb, channels="RGB", use_container_width=True),
        stats_callback=lambda el, dur, reps, low_conf, last_score: stats_box.markdown(
            f"**Elapsed:** {el:.1f}s / {dur}s | "
            f"**Detected reps:** {reps} | "
            f"**Low-confidence frames:** {low_conf} | "
            f"**Last rep score:** {('--' if last_score is None else int(round(last_score)))}"
        ),
        status_callback=status_callback,
    )


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
    st.markdown(
        '<div class="step-card"><h3>Step 1: Input</h3><p>Pick a source, verify preview, then run analysis.</p></div>',
        unsafe_allow_html=True,
    )

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
    st.markdown(
        '<div class="step-card"><h3>Step 2: Output</h3><p>Review metrics, top issues, and actionable feedback.</p></div>',
        unsafe_allow_html=True,
    )

    if run_clicked or demo_clicked:
        if source_mode == MODE_LIVE:
            with st.spinner("Starting webcam assessment..."):
                report = run_streamlit_live(
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
