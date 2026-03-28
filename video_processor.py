import cv2
import numpy as np
import sys

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


# ================= VIDEO =================
def process_video(video_path, output_path, exercise="pushup"):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
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

    # ================= SUMMARY =================
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

    elif exercise == "squat" and squat_history:
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
                print(f"  • {k.replace('_', ' ').title()}: {v}/{len(squat_history)} reps")

    print("\n==============================")
    print(f"Total reps: {reps}")
    print("==============================")


# ================= WEBCAM =================
def process_webcam(exercise="pushup"):
    cap = cv2.VideoCapture(0)
    estimator = PoseEstimator()

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
    last_feedback = []

    print(f"Starting webcam - {exercise}. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        stage = "N/A"
        live_msg = ""
        results = estimator.process_frame(frame)
        landmarks = estimator.get_landmarks(results, frame.shape)
        angles = get_all_angles(landmarks)

        if landmarks:
            if exercise == "pushup":
                angle = get_elbow(landmarks, angles)
            else:
                angle = angles.get('left_knee') or angles.get('right_knee')

            smooth_angle = temporal.smooth(angle)

            if smooth_angle is not None:
                stage = temporal.detect_stage(smooth_angle)

            # ===== PUSHUP =====
            if exercise == "pushup":
                analyzer.collect(landmarks, angles)

                if counter.update(smooth_angle):
                    reps += 1

                    feedback_lines = analyzer.evaluate()

                    score = 0
                    issues = []

                    for line in feedback_lines:
                        if "Quality:" in line:
                            score = int(line.split("Quality:")[1].split("/")[0])
                        if "[ISSUE]" in line:
                            issues.append(line)

                    last_feedback = [f"Score: {score}/100"]

                    for line in feedback_lines:
                        if "[GOOD]" in line:
                            last_feedback.append(line.replace("[GOOD] ", ""))
                            break

                    for issue in issues[:2]:
                        last_feedback.append(issue.replace("[ISSUE] ", ""))

                    analyzer.reset()

            # ===== SQUAT =====
            else:
                if smooth_angle is not None and counter.update(smooth_angle, landmarks):
                    reps += 1
                    rep_angles = counter.get_last_rep()
                    landmarks_seq = counter.get_last_landmarks()

                    result = squat_analyzer.analyze(rep_angles, landmarks_seq)

                    last_feedback = [f"Score: {result['score']}/100"]

                    for fb in result["feedback"]:
                        last_feedback.append(fb)

                    for issue in result["issues"][:2]:
                        last_feedback.append(issue.replace("_", " ").title())

        frame = estimator.draw_skeleton(frame, results)

        cv2.putText(frame, f"{exercise.upper()}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.putText(frame, f"Reps: {reps}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Stage: {stage}", (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if live_msg:
            cv2.putText(frame, live_msg, (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        cv2.rectangle(frame, (10, 110), (450, 250), (0, 0, 0), -1)

        y = 140
        for msg in last_feedback[:3]:
            color = (0, 255, 0) if ("Good" in msg or "Excellent" in msg or "Score" in msg) else (0, 0, 255)

            cv2.putText(frame, msg, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 30

        cv2.imshow("FormAI Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()