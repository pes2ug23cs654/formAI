import streamlit as st
from components.sidebar import sidebar
from components.uploader import video_uploader
from components.feedback import show_feedback
import sys
import os
import tempfile
import time
import subprocess


def save_uploaded_file(uploaded_file):
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def transcode_for_web(input_path):
    """Transcode to H.264 + faststart so browser players can decode reliably."""
    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        output_path = os.path.splitext(input_path)[0] + "_web.mp4"

        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            input_path,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            output_path,
        ]

        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path, None

        return input_path, (completed.stderr or completed.stdout or "Unknown ffmpeg error")
    except Exception as exc:
        return input_path, str(exc)

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import run_analysis
from video_processor import process_video

st.set_page_config(page_title="AI Form Analyzer", layout="wide")

# Load custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Sidebar
exercise = sidebar()

# Title
st.title("🏋️ FormAI")

# Upload Section
video_file = video_uploader()

if video_file is not None:
    video_path = save_uploaded_file(video_file)
    output_video_path = os.path.join(tempfile.gettempdir(), f"formAI_annotated_{int(time.time()*1000)}.mp4")
    web_video_path = output_video_path
    transcode_error = None

    with st.spinner("Processing video and generating annotations..."):
        process_video(video_path, output_video_path, exercise.lower())
        time.sleep(0.5)
        web_video_path, transcode_error = transcode_for_web(output_video_path)
        results = run_analysis(video_path, exercise.lower())

    st.subheader("🎥 Annotated Output")
    if os.path.exists(web_video_path) and os.path.getsize(web_video_path) > 0:
        file_size_mb = os.path.getsize(web_video_path) / (1024 * 1024)
        st.info(f"✓ Annotated video ready ({file_size_mb:.1f} MB)")
        if transcode_error:
            st.warning("Used original codec because web transcode failed.")
        with open(web_video_path, "rb") as f:
            st.video(f.read(), format="video/mp4")
    else:
        file_size = os.path.getsize(web_video_path) if os.path.exists(web_video_path) else 0
        st.error(f"Annotated output video could not be generated. File size: {file_size} bytes. Path: {web_video_path}")

    st.subheader("📊 Analysis Results")

    show_feedback(results)