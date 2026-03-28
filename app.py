"""FormAI entrypoint with CLI + Streamlit UI"""

import argparse
from pathlib import Path

from video_processor import process_video
from src.exercise_profiles import list_exercise_keys


def run_cli():
    parser = argparse.ArgumentParser(
        description="FormAI: Analyze exercise form from a video."
    )

    parser.add_argument(
        "--input", "-i",
        default="input/input.mp4",
        help="Input video path"
    )

    parser.add_argument(
        "--output", "-o",
        default="output/output.mp4",
        help="Output annotated video"
    )

    parser.add_argument(
        "--report-json", "-r",
        default="report/report.json",
        help="Path to save JSON report"
    )

    parser.add_argument(
        "--exercise", "-e",
        choices=list_exercise_keys(),
        default="pushup"
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.55
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report_json)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = process_video(
        str(input_path),
        str(output_path),
        report_json_path=str(report_path),
        exercise=args.exercise,
        confidence_threshold=args.confidence_threshold,
    )

    print("\nRun complete")
    print(f"Output video: {output_path}")
    print(f"Report JSON: {report_path}")
    print(f"Total reps: {report['summary']['total_reps']}")


# ---------------- STREAMLIT UI ---------------- #

def run_streamlit():
    import streamlit as st

    st.title("🏋️ FormAI Exercise Analyzer")

    exercise = st.selectbox(
        "Select Exercise",
        list_exercise_keys()
    )

    uploaded_video = st.file_uploader(
        "Upload exercise video",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_video:
        input_path = Path("input/uploaded.mp4")
        output_path = Path("output/output.mp4")
        report_path = Path("report/report.json")

        input_path.parent.mkdir(exist_ok=True)
        output_path.parent.mkdir(exist_ok=True)
        report_path.parent.mkdir(exist_ok=True)

        with open(input_path, "wb") as f:
            f.write(uploaded_video.read())

        if st.button("Analyze Exercise"):
            with st.spinner("Processing video..."):
                report = process_video(
                    str(input_path),
                    str(output_path),
                    report_json_path=str(report_path),
                    exercise=exercise,
                )

            st.success("Analysis Complete")

            st.video(str(output_path))

            st.subheader("Results")
            st.write(f"Total Reps: {report['summary']['total_reps']}")
            st.write(f"Valid Reps: {report['summary']['valid_reps']}")
            st.write(f"Invalid Reps: {report['summary']['invalid_reps']}")
            st.write(f"Average Score: {report['summary']['avg_score']}")


# ---------------- ENTRYPOINT ---------------- #

if __name__ == "__main__":
    import sys

    if "streamlit" in sys.argv[0]:
        run_streamlit()
    else:
        run_cli()