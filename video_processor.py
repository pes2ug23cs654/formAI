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

            # DEBUG
            if frame_count % 50 == 0 and smooth_angle is not None:
                left = angles.get('left_elbow')
                right = angles.get('right_elbow')
                print(f"[F{frame_count}] Angle:{smooth_angle:.1f}° (L:{left} R:{right}) | State:{counter.state}")

            analyzer.collect(landmarks, angles)

            if counter.update(smooth_angle):
                reps += 1

                from src.form_standards import assess_injury_risk, get_form_quality_score

                elbows = [f['elbow'] for f in analyzer.rep_angles if f['elbow']]
                hips = [f['hip'] for f in analyzer.rep_angles if f['hip']]

                analysis_angles = {
                    'left_elbow': min(elbows) if elbows else None,
                    'left_hip': min(hips) if hips else None,
                }

                risk = assess_injury_risk(analysis_angles, None)
                score = get_form_quality_score(risk)

                session.add_rep({
                    'rep_num': reps,
                    'score': score,
                    'issues': risk.get('issues', []),
                    'feedback': risk.get('feedback', [])
                })

                print(f"\n[REP {reps}] Score: {score}/100")

                positives = [f for f in risk['feedback'] if "[GOOD]" in f]
                if positives:
                    print(f"  ✓ {positives[0].replace('[GOOD] ', '')}")
                else:
                    print("  ✓ Form acceptable")

                for issue in risk['issues'][:2]:
                    print(f"  ⚠ {issue.replace('_', ' ').title()}")

                analyzer.reset()

        # ================= SQUATS =================
        else:
            angle = angles.get('left_knee') or angles.get('right_knee')
            smooth_angle = temporal.smooth(angle)

            if smooth_angle is not None:
                stage = temporal.detect_stage(smooth_angle)

            if frame_count % 50 == 0 and smooth_angle is not None:
                print(f"[F{frame_count}] Knee:{smooth_angle:.1f}° | State:{counter.state}")

            if counter.update(smooth_angle,landmarks):
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
                else:
                    print(f"\n[SQUAT {reps}] Completed")

        # ================= DRAW =================
        frame = estimator.draw_skeleton(frame, results)

        cv2.putText(frame, f"{exercise.upper()}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(frame, f"Reps: {reps}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Stage: {stage}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

# ================= SUMMARY =================

# ===== PUSHUP SUMMARY =====
    if exercise == "pushup":
        report = session.analyze()

        print("\n" + "="*50)
        print(f"SESSION SUMMARY - {reps} Reps Completed")
        print("="*50)

    # SAFE SCORE
        avg_score = report.get("avg_score") or report.get("average_score") or report.get("overall_score")

        if avg_score is None:
            scores = []
            if hasattr(session, "reps"):
                scores = [r.get("score", 0) for r in session.reps]
            avg_score = sum(scores) // len(scores) if scores else 0

        print(f"Overall Quality: {avg_score}/100")

    # ISSUES
        if report.get("issues"):
            print("\n🎯 PRIMARY FORM ISSUES:")
            for issue, count in report["issues"].items():
                print(f"  • {issue}: {count}/{reps} reps")

    # TIPS
        print("\n💡 COACH TIPS:")
        for tip in report.get("feedback", []):
            print(f"  → {tip}")

    # ANGLES
        all_angles = [f['elbow'] for f in analyzer.rep_angles if f['elbow']]
        if all_angles:
            print(f"\nAngle range: {min(all_angles):.1f}° - {max(all_angles):.1f}°")


# ===== SQUAT SUMMARY =====
    elif exercise == "squat" and squat_history:
        print("\n" + "="*50)
        print(f"SESSION SUMMARY - {reps} Reps Completed")
        print("="*50)

        avg_score = sum(r["score"] for r in squat_history) // len(squat_history)
        print(f"Overall Quality: {avg_score}/100")

        issue_count = {}
        for rep in squat_history:
            for issue in rep["issues"]:
                issue_count[issue] = issue_count.get(issue, 0) + 1

        if issue_count:
            print("\n🎯 PRIMARY ISSUES:")
            for k, v in issue_count.items():
                readable = k.replace('_', ' ').title()
                print(f"  • {readable}: {v}/{len(squat_history)} reps")
        print("\n💡 COACH TIP:")
        if not issue_count:
            print("  → Excellent form! Maintain consistency.")
        else:
            if "shallow_squat" in issue_count:
                print("  → Go deeper (target <90° knee angle)")
            if "poor_hip_hinge" in issue_count:
                print("  → Push hips back more")
            if "back_rounding" in issue_count:
                print("  → Keep chest up and spine neutral")
            print(f"\nTotal reps analyzed: {len(squat_history)}")


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

                    from src.form_standards import assess_injury_risk, get_form_quality_score

                    elbows = [f['elbow'] for f in analyzer.rep_angles if f['elbow']]
                    hips = [f['hip'] for f in analyzer.rep_angles if f['hip']]

                    analysis_angles = {
                        'left_elbow': min(elbows) if elbows else None,
                        'left_hip': min(hips) if hips else None,
                    }

                    risk = assess_injury_risk(analysis_angles, None)
                    score = get_form_quality_score(risk)

                    last_feedback = [f"Score: {score}/100"]

                    positives = [f for f in risk['feedback'] if "[GOOD]" in f]
                    if positives:
                        last_feedback.append(positives[0].replace("[GOOD] ", ""))
                    else:
                        last_feedback.append("Form acceptable")

                    for issue in risk['issues'][:2]:
                        last_feedback.append(issue.replace("_", " ").title())

                    analyzer.reset()

            # ===== SQUAT =====
            else:
                if counter.update(smooth_angle, landmarks):
                    reps += 1
                    rep_angles = counter.get_last_rep()
                    landmarks_seq = counter.get_last_landmarks()

                    result = squat_analyzer.analyze(rep_angles, landmarks_seq)

                    last_feedback = [f"Score: {result['score']}/100"]

                    for fb in result["feedback"]:
                        last_feedback.append(fb)

                    for issue in result["issues"][:2]:
                        last_feedback.append(issue.replace("_", " ").title())

        # DRAW
        frame = estimator.draw_skeleton(frame, results)

        cv2.putText(frame, f"{exercise.upper()}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.putText(frame, f"Reps: {reps}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Stage: {stage}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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