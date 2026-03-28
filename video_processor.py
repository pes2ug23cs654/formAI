import cv2
import numpy as np
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src.squat_analyzer import SquatAnalyzer
from src.pose_estimator import PoseEstimator
from src.angle_engine import get_all_angles, get_visibility, LANDMARKS
from src.temporal_engine import TemporalEngine
from src.rep_counter import RepCounter
from src.form_analyzer import FormAnalyzer
from src.session_analyzer import SessionAnalyzer


# ================= HELPER =================
def get_elbow(landmarks, angles):
    left_visible = get_visibility(landmarks, [
        LANDMARKS['left_shoulder'],
        LANDMARKS['left_elbow'],
        LANDMARKS['left_wrist']
    ])

    right_visible = get_visibility(landmarks, [
        LANDMARKS['right_shoulder'],
        LANDMARKS['right_elbow'],
        LANDMARKS['right_wrist']
    ])

    if left_visible:
        return angles.get('left_elbow')
    elif right_visible:
        return angles.get('right_elbow')
    return None


def _ffmpeg_executable():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def reencode_for_html5_video(src_path: str, dst_path: str) -> bool:
    """
    Browsers usually cannot play OpenCV's MPEG-4 ('mp4v') output. Re-encode to
    H.264 (yuv420p) + faststart so <video> can play in Chrome/Edge/Firefox.
    """
    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        print(
            "Warning: ffmpeg not found (install ffmpeg or `pip install imageio-ffmpeg`). "
            "Output may not play in web browsers."
        )
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        src_path,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        dst_path,
    ]
    kwargs = {}
    if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    r = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if r.returncode != 0:
        print(f"ffmpeg re-encode failed: {r.stderr or r.stdout}")
        return False
    return True


# ================= VIDEO =================
def process_video(video_path, output_path, exercise="pushup", web_playback=False):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if web_playback:
        fd, cv2_tmp = tempfile.mkstemp(suffix=".mp4", dir=str(out_path.parent))
        os.close(fd)
        writer_target = cv2_tmp
    else:
        writer_target = str(out_path)

    out = cv2.VideoWriter(
        writer_target,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    estimator = PoseEstimator()
    session = SessionAnalyzer()

    print(f"Running exercise: {exercise}")

    if exercise == "pushup":
        temporal = TemporalEngine()
        counter = RepCounter()
        analyzer = FormAnalyzer()
    else:
        from src.squat_rep_counter import SquatRepCounter
        from src.squat_temporal_engine import SquatTemporalEngine

        temporal = SquatTemporalEngine()
        counter = SquatRepCounter()
        squat_analyzer = SquatAnalyzer()

    reps = 0
    frame_count = 0
    squat_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        stage = "N/A"
        live_msg = ""

        results = estimator.process_frame(frame)
        landmarks = estimator.get_landmarks(results, frame.shape)
        angles = get_all_angles(landmarks)

        if landmarks is None:
            out.write(frame)
            continue

        # ================= PUSHUPS =================
        if exercise == "pushup":
            angle = get_elbow(landmarks, angles)
            smooth_angle = temporal.smooth(angle)

            if smooth_angle is not None:
                stage = temporal.detect_stage(smooth_angle)

            if frame_count % 50 == 0 and smooth_angle is not None:
                print(f"[F{frame_count}] Angle:{smooth_angle:.1f}° | State:{counter.state}")

            analyzer.collect(landmarks, angles)

            if smooth_angle is not None and counter.update(smooth_angle):
                reps += 1

                feedback_lines = analyzer.evaluate()

                score = 0
                issues = []

                for line in feedback_lines:
                    if "Quality:" in line:
                        score = int(line.split("Quality:")[1].split("/")[0])
                    if "[ISSUE]" in line:
                        issues.append(line)

                session.add_rep({
                    'rep_num': reps,
                    'score': score,
                    'issues': issues,
                    'feedback': feedback_lines
                })

                print("\n".join(feedback_lines))
                analyzer.reset()

        # ================= SQUATS =================
        else:
            angle = angles.get('left_knee') or angles.get('right_knee')
            smooth_angle = temporal.smooth(angle)

            if smooth_angle is not None:
                stage = temporal.detect_stage(smooth_angle)

                # ===== REAL-TIME COACHING =====
                shoulder = landmarks[11]
                hip = landmarks[23]
                knee = landmarks[25]
                ankle = landmarks[27]

                # Depth
                if smooth_angle > 100:
                    live_msg = "Go deeper"

                # Back posture
                dx = shoulder['x'] - hip['x']
                dy = shoulder['y'] - hip['y']
                back_angle = abs(np.degrees(np.arctan2(dy, dx)))

                if back_angle < 50:
                    live_msg = "Keep chest up"

                # Hip hinge
                body_length = np.linalg.norm([
                    hip['x'] - shoulder['x'],
                    hip['y'] - shoulder['y']
                ])

                hip_shift = abs(hip['x'] - ankle['x'])
                ratio = hip_shift / (body_length + 1e-6)

                if ratio < 0.15:
                    live_msg = "Push hips back"

            # DEBUG
            if frame_count % 50 == 0 and smooth_angle is not None:
                print(f"[F{frame_count}] Knee:{smooth_angle:.1f}° | State:{counter.state}")

            # ✅ REP DETECTION FIXED
            if smooth_angle is not None and counter.update(smooth_angle, landmarks):
                reps += 1
                rep_angles = counter.get_last_rep()

                if rep_angles:
                    landmarks_seq = counter.get_last_landmarks()
                    result = squat_analyzer.analyze(rep_angles, landmarks_seq)

                    print(f"\n[SQUAT {reps}] Score: {result['score']}/100")
                    print(f"  → Knee Angle: {result['knee_angle']:.1f}°")

                    for fb in result["feedback"]:
                        print(f"  ✓ {fb}")

                    for issue in result["issues"][:3]:
                        print(f"  ⚠ {issue.replace('_', ' ').title()}")

                    squat_history.append({
                        "score": result["score"],
                        "issues": result["issues"]
                    })

        # ================= DRAW =================
        frame = estimator.draw_skeleton(frame, results)

        cv2.putText(frame, f"{exercise.upper()}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(frame, f"Reps: {reps}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Stage: {stage}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if live_msg:
            cv2.putText(frame, live_msg, (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        out.write(frame)

    cap.release()
    out.release()

    if web_playback:
        fd, h264_tmp = tempfile.mkstemp(suffix=".mp4", dir=str(out_path.parent))
        os.close(fd)
        try:
            if reencode_for_html5_video(writer_target, h264_tmp):
                os.replace(h264_tmp, str(out_path))
                os.unlink(writer_target)
            else:
                try:
                    os.unlink(h264_tmp)
                except OSError:
                    pass
                shutil.move(writer_target, str(out_path))
        except OSError as e:
            print(f"Warning: could not finalize web-playable video: {e}")
            try:
                os.unlink(h264_tmp)
            except OSError:
                pass
            if os.path.isfile(writer_target):
                shutil.move(writer_target, str(out_path))

    # ================= SUMMARY =================
    coach = {"reps": reps, "feedback_lines": []}

    if exercise == "pushup":
        report = session.analyze()

        print("\n" + "=" * 50)
        print(f"SESSION SUMMARY - {reps} Reps Completed")
        print("=" * 50)

        avg_score = report.get("overall_score", 0)
        print(f"Overall Quality: {avg_score}/100")

        if report.get("primary_issues"):
            print("\n🎯 PRIMARY FORM ISSUES:")
            for issue, count in report["primary_issues"]:
                print(f"  • {issue}: {count}/{reps} reps")

        print("\n💡 COACH FEEDBACK:")
        for line in report.get("feedback", []):
            print(line)

        if reps == 0:
            coach["feedback_lines"] = [
                "No push-up reps detected. Try a clearer side or 45° angle, full body in frame, and enough reps in the clip.",
            ]
        else:
            coach["feedback_lines"] = [
                f"Score: {avg_score}/100",
                f"Session: {reps} rep(s) in this video.",
            ]
            for issue, count in report.get("primary_issues") or []:
                coach["feedback_lines"].append(
                    f"{issue.replace('_', ' ').title()}: {count}/{reps} reps"
                )
            if not report.get("primary_issues"):
                coach["feedback_lines"].append(
                    "No recurring issues flagged — solid session."
                )

    elif exercise == "squat":
        if squat_history:
            print("\n" + "=" * 50)
            print(f"SESSION SUMMARY - {reps} Reps Completed")
            print("=" * 50)

            avg_score = sum(r["score"] for r in squat_history) // len(squat_history)
            print(f"Overall Quality: {avg_score}/100")

            issue_count = {}
            for rep in squat_history:
                for issue in rep["issues"]:
                    issue_count[issue] = issue_count.get(issue, 0) + 1

            if issue_count:
                print("\n🎯 PRIMARY ISSUES:")
                for k, v in issue_count.items():
                    print(
                        f"  • {k.replace('_', ' ').title()}: {v}/{len(squat_history)} reps"
                    )

            coach["feedback_lines"] = [
                f"Score: {avg_score}/100",
                f"Session: {len(squat_history)} squat rep(s) in this video.",
            ]
            for k, v in sorted(issue_count.items(), key=lambda x: -x[1])[:5]:
                coach["feedback_lines"].append(
                    f"{k.replace('_', ' ').title()}: {v} rep(s)"
                )
            if not issue_count:
                coach["feedback_lines"].append(
                    "No recurring issues flagged — solid session."
                )
        elif reps == 0:
            coach["feedback_lines"] = [
                "No squat reps detected. Face the camera, keep full body visible, and show a clear down–up motion.",
            ]
        else:
            coach["feedback_lines"] = [
                f"Session: {reps} rep(s) detected.",
                "Detailed scores were not available for these reps (pose visibility).",
            ]

    print("\n==============================")
    print(f"Total reps: {reps}")
    print("==============================")

    return coach


# ================= LIVE (WEBCAM / UI) =================
class LivePipeline:
    """Single-session pose pipeline: one frame in, annotated frame + metadata out."""

    def __init__(self, exercise="pushup"):
        self.exercise = exercise
        self.estimator = PoseEstimator()
        self.reps = 0
        self.last_feedback = []

        if exercise == "pushup":
            self.temporal = TemporalEngine()
            self.counter = RepCounter()
            self.analyzer = FormAnalyzer()
            self.squat_analyzer = None
        else:
            from src.squat_rep_counter import SquatRepCounter
            from src.squat_temporal_engine import SquatTemporalEngine

            self.temporal = SquatTemporalEngine()
            self.counter = SquatRepCounter()
            self.analyzer = None
            self.squat_analyzer = SquatAnalyzer()

    def process(self, frame):
        stage = "N/A"
        live_msg = ""
        smooth_angle = None

        results = self.estimator.process_frame(frame)
        landmarks = self.estimator.get_landmarks(results, frame.shape)
        angles = get_all_angles(landmarks)

        if landmarks:
            if self.exercise == "pushup":
                angle = get_elbow(landmarks, angles)
            else:
                angle = angles.get("left_knee") or angles.get("right_knee")

            smooth_angle = self.temporal.smooth(angle)

            if smooth_angle is not None:
                stage = self.temporal.detect_stage(smooth_angle)

            if self.exercise == "pushup":
                self.analyzer.collect(landmarks, angles)

                if self.counter.update(smooth_angle):
                    self.reps += 1

                    feedback_lines = self.analyzer.evaluate()

                    score = 0
                    issues = []

                    for line in feedback_lines:
                        if "Quality:" in line:
                            score = int(line.split("Quality:")[1].split("/")[0])
                        if "[ISSUE]" in line:
                            issues.append(line)

                    self.last_feedback = [f"Score: {score}/100"]

                    for line in feedback_lines:
                        if "[GOOD]" in line:
                            self.last_feedback.append(line.replace("[GOOD] ", ""))
                            break

                    for issue in issues[:2]:
                        self.last_feedback.append(issue.replace("[ISSUE] ", ""))

                    print("\n".join(feedback_lines))
                    self.analyzer.reset()
            else:
                if smooth_angle is not None:
                    shoulder = landmarks[11]
                    hip = landmarks[23]
                    knee = landmarks[25]
                    ankle = landmarks[27]

                    if smooth_angle > 100:
                        live_msg = "Go deeper"

                    dx = shoulder["x"] - hip["x"]
                    dy = shoulder["y"] - hip["y"]
                    back_angle = abs(np.degrees(np.arctan2(dy, dx)))

                    if back_angle < 50:
                        live_msg = "Keep chest up"

                    body_length = np.linalg.norm(
                        [hip["x"] - shoulder["x"], hip["y"] - shoulder["y"]]
                    )
                    hip_shift = abs(hip["x"] - ankle["x"])
                    ratio = hip_shift / (body_length + 1e-6)

                    if ratio < 0.15:
                        live_msg = "Push hips back"

                if smooth_angle is not None and self.counter.update(
                    smooth_angle, landmarks
                ):
                    self.reps += 1
                    rep_angles = self.counter.get_last_rep()
                    landmarks_seq = self.counter.get_last_landmarks()

                    result = self.squat_analyzer.analyze(rep_angles, landmarks_seq)

                    self.last_feedback = [f"Score: {result['score']}/100"]

                    for fb in result["feedback"]:
                        self.last_feedback.append(fb)

                    for issue in result["issues"][:2]:
                        self.last_feedback.append(issue.replace("_", " ").title())

                    print(f"\n[SQUAT {self.reps}] Score: {result['score']}/100")
                    for fb in result["feedback"]:
                        print(f"  ✓ {fb}")
                    for issue in result["issues"][:3]:
                        print(f"  ⚠ {issue.replace('_', ' ').title()}")

        frame = self.estimator.draw_skeleton(frame, results)

        cv2.putText(
            frame,
            f"{self.exercise.upper()}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Reps: {self.reps}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Stage: {stage}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        if live_msg:
            cv2.putText(
                frame,
                live_msg,
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
            )

        cv2.rectangle(frame, (10, 110), (450, 250), (0, 0, 0), -1)

        y = 140
        for msg in self.last_feedback[:3]:
            color = (
                (0, 255, 0)
                if ("Good" in msg or "Excellent" in msg or "Score" in msg)
                else (0, 0, 255)
            )

            cv2.putText(
                frame, msg, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
            y += 30

        meta = {
            "reps": self.reps,
            "stage": stage,
            "live_msg": live_msg,
            "feedback_lines": list(self.last_feedback[:5]),
            "smooth_angle": None if smooth_angle is None else float(smooth_angle),
        }
        return frame, meta


def process_webcam(exercise="pushup"):
    cap = cv2.VideoCapture(0)
    pipeline = LivePipeline(exercise)

    print(f"Starting webcam - {exercise}. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, _meta = pipeline.process(frame)

        cv2.imshow("FormAI Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()