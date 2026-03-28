"""CLI entrypoint for FormAI video processing."""

import argparse
from pathlib import Path

from video_processor import process_video
from src.exercise_profiles import list_exercise_keys


def build_parser():
	parser = argparse.ArgumentParser(
		description="FormAI: Analyze exercise form from a video."
	)
	parser.add_argument(
		"--input",
		"-i",
		default="input/input.mp4",
		help="Input video path (default: input/input.mp4)",
	)
	parser.add_argument(
		"--output",
		"-o",
		default="output/output.mp4",
		help="Output annotated video path (default: output/output.mp4)",
	)
	parser.add_argument(
		"--report-json",
		"-r",
		default="report/report.json",
		help="Path to write structured JSON report (default: report/report.json)",
	)
	parser.add_argument(
		"--quiet",
		action="store_true",
		help="Disable debug logs.",
	)
	parser.add_argument(
		"--calibration-seconds",
		type=int,
		default=10,
		help="Seconds used for threshold calibration (default: 10).",
	)
	parser.add_argument(
		"--exercise",
		"-e",
		choices=list_exercise_keys(),
		default="pushup",
		help="Exercise type to analyze.",
	)
	parser.add_argument(
		"--confidence-threshold",
		type=float,
		default=0.55,
		help="Minimum landmark confidence for rep logic (default: 0.55).",
	)
	return parser


def main():
	parser = build_parser()
	args = parser.parse_args()

	input_path = Path(args.input)
	output_path = Path(args.output)
	report_path = Path(args.report_json)

	if not input_path.exists():
		raise FileNotFoundError(f"Input video not found: {input_path}")

	if output_path.parent and not output_path.parent.exists():
		output_path.parent.mkdir(parents=True, exist_ok=True)

	if report_path.parent and not report_path.parent.exists():
		report_path.parent.mkdir(parents=True, exist_ok=True)

	report = process_video(
		str(input_path),
		str(output_path),
		report_json_path=str(report_path),
		debug=not args.quiet,
		calibration_seconds=args.calibration_seconds,
		confidence_threshold=args.confidence_threshold,
		exercise=args.exercise,
	)

	print("\nRun complete")
	print(f"Output video: {output_path}")
	print(f"Report JSON: {report_path}")
	print(
		"Summary: "
		f"exercise={report['exercise']} "
		f"reps={report['summary']['total_reps']} "
		f"valid={report['summary']['valid_reps']} "
		f"invalid={report['summary']['invalid_reps']} "
		f"avg_score={report['summary']['avg_score']}"
	)


if __name__ == "__main__":
	main()
