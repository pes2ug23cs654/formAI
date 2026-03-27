import cv2
import numpy as np
import sys

from src.pose_estimator import PoseEstimator
from src.angle_engine import get_all_angles, get_visibility, LANDMARKS
from src.temporal_engine import TemporalEngine
from src.rep_counter import RepCounter
from src.rep_analyzer import RepAnalyzer
from src.form_analyzer import FormAnalyzer
from src.session_analyzer import SessionAnalyzer
from src.thresholds import TOP_ANGLE, BOTTOM_ANGLE, DESCENT_TRIGGER, ASCENT_THRESHOLD


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


def process_video(video_path, output_path):

    cap = cv2.VideoCapture(video_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    estimator = PoseEstimator()
    temporal = TemporalEngine()
    counter = RepCounter()
    analyzer = FormAnalyzer()
    session = SessionAnalyzer()  # Aggregate all reps

    reps = 0
    last_feedback = []
    
    frame_count = 0
    angle_samples = []  # DEBUG

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = estimator.process_frame(frame)
        landmarks = estimator.get_landmarks(results, frame.shape)
        angles = get_all_angles(landmarks)

        if landmarks is None:
            out.write(frame)
            continue

        elbow = get_elbow(landmarks, angles)

        # TRACK MOTION
        smooth_angle = temporal.smooth(elbow)
        stage = temporal.detect_stage(smooth_angle)
        
        # DEBUG: Sample angles every 50 frames (more frequent)
        if frame_count % 50 == 0 and smooth_angle:
            angle_samples.append(smooth_angle)
            # Show which arm is being tracked
            left_elbow = angles.get('left_elbow')
            right_elbow = angles.get('right_elbow')
            print(f"[F{frame_count:3d}] Angle:{smooth_angle:6.1f}° (L:{left_elbow or 'N/A':>5} R:{right_elbow or 'N/A':>5}) | State:{counter.state}", flush=True)

        # ALWAYS COLLECT
        analyzer.collect(landmarks, angles)

        # COUNT REPS (assume already in plank)
        if counter.update(smooth_angle):
            reps += 1

            feedback = analyzer.evaluate()
            
            # Extract structured data for session analysis
            from src.form_standards import assess_injury_risk, get_form_quality_score
            
            elbows = [f['elbow'] for f in analyzer.rep_angles if f['elbow']]
            hips = [f['hip'] for f in analyzer.rep_angles if f['hip']]
            
            analysis_angles = {
                'left_elbow': min(elbows) if elbows else None,
                'left_hip': min(hips) if hips else None,
            }
            
            # Extract bottom landmarks
            bottom_landmarks = None
            if elbows:
                min_elbow_idx = elbows.index(min(elbows))
                bottom_landmarks = analyzer.rep_angles[min_elbow_idx]['landmarks']
            
            risk_assessment = assess_injury_risk(analysis_angles, bottom_landmarks)
            score = get_form_quality_score(risk_assessment)
            
            rep_data = {
                'rep_num': reps,
                'score': score,
                'issues': risk_assessment.get('issues', []),
                'flags': risk_assessment.get('flags', []),
                'feedback': feedback
            }
            
            session.add_rep(rep_data)
            
            # ─── BRIEF PER-REP FEEDBACK ────────────────────────
            print(f"\n[REP {reps}] Score: {score}/100", flush=True)
            
            # Show positives first
            positives = [f for f in risk_assessment['feedback'] if "[GOOD]" in f]
            if positives:
                print(f"  ✓ {positives[0].replace('[GOOD] ', '')}", flush=True)
            else:
                print(f"  ✓ Form acceptable", flush=True)
            
            # Show issues if any
            if risk_assessment['issues']:
                for issue in risk_assessment['issues'][:2]:  # Show top 2 issues
                    print(f"  ⚠ {issue.replace('_', ' ').title()}", flush=True)
            
            last_feedback = [f"Rep {reps}: {score}/100"]

            analyzer.reset()

            analyzer.reset()

        # DRAW
        frame = estimator.draw_skeleton(frame, results)

        # UI
        cv2.putText(frame, f"Reps: {reps}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, f"Stage: {stage}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        y = 120
        for msg in last_feedback:
            cv2.putText(frame, msg, (20,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            y += 30

        out.write(frame)

    # ─── FINALIZE IN-PROGRESS REP ────────────────────────
    if counter.state == "DOWN" and len(counter.angles_in_rep) > 10:
        if counter.update(None):  # Finalize video-end rep
            reps += 1
            analyzer.collect(None, None)  # Empty collect to mark completion
            try:
                feedback = analyzer.evaluate()
                
                # Extract structured data
                from src.form_standards import assess_injury_risk, get_form_quality_score
                
                elbows = [f['elbow'] for f in analyzer.rep_angles if f['elbow']]
                hips = [f['hip'] for f in analyzer.rep_angles if f['hip']]
                
                analysis_angles = {
                    'left_elbow': min(elbows) if elbows else None,
                    'left_hip': min(hips) if hips else None,
                }
                
                bottom_landmarks = None
                if elbows:
                    min_elbow_idx = elbows.index(min(elbows))
                    bottom_landmarks = analyzer.rep_angles[min_elbow_idx]['landmarks']
                
                risk_assessment = assess_injury_risk(analysis_angles, bottom_landmarks)
                score = get_form_quality_score(risk_assessment)
                
                # Show brief feedback for final rep
                print(f"\n[REP {reps}] Score: {score}/100", flush=True)
                positives = [f for f in risk_assessment['feedback'] if "[GOOD]" in f]
                if positives:
                    print(f"  ✓ {positives[0].replace('[GOOD] ', '')}", flush=True)
                else:
                    print(f"  ✓ Form acceptable", flush=True)
                
                if risk_assessment['issues']:
                    for issue in risk_assessment['issues'][:2]:
                        print(f"  ⚠ {issue.replace('_', ' ').title()}", flush=True)
                
                rep_data = {
                    'rep_num': reps,
                    'score': score,
                    'issues': risk_assessment.get('issues', []),
                    'flags': risk_assessment.get('flags', []),
                    'feedback': feedback
                }
                
                session.add_rep(rep_data)
                
            except:
                pass

    cap.release()
    out.release()

    # ═══════════════════════════════════════════════════════════
    # GENERATE CONSOLIDATED SESSION REPORT
    # ═══════════════════════════════════════════════════════════
    session_report = session.analyze()
    
    for line in session_report['feedback']:
        print(line, flush=True)
    
    print(f"\n{'='*60}")
    print(f"Total reps: {reps}")
    print(f"{'='*60}")
    
    # DEBUG: Show angle details
    if angle_samples:
        print(f"Angle range: {min(angle_samples):.1f}° - {max(angle_samples):.1f}°")
        print(f"Avg angle: {sum(angle_samples)/len(angle_samples):.1f}°")


def process_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    estimator = PoseEstimator()
    temporal = TemporalEngine()
    counter = RepCounter()
    analyzer = FormAnalyzer()
    session = SessionAnalyzer()  # Aggregate all reps

    reps = 0
    last_feedback = []
    
    frame_count = 0
    angle_samples = []  # DEBUG

    print("Starting real-time webcam analysis. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = estimator.process_frame(frame)
        landmarks = estimator.get_landmarks(results, frame.shape)
        angles = get_all_angles(landmarks)

        if landmarks is not None:
            elbow = get_elbow(landmarks, angles)

            # TRACK MOTION
            smooth_angle = temporal.smooth(elbow)
            stage = temporal.detect_stage(smooth_angle)
            
            # DEBUG: Sample angles every 50 frames (more frequent)
            if frame_count % 50 == 0 and smooth_angle:
                angle_samples.append(smooth_angle)
                # Show which arm is being tracked
                left_elbow = angles.get('left_elbow')
                right_elbow = angles.get('right_elbow')
                print(f"[F{frame_count:3d}] Angle:{smooth_angle:6.1f}° (L:{left_elbow or 'N/A':>5} R:{right_elbow or 'N/A':>5}) | State:{counter.state}", flush=True)

            # ALWAYS COLLECT
            analyzer.collect(landmarks, angles)

            # COUNT REPS (assume already in plank)
            if counter.update(smooth_angle):
                reps += 1
                feedback = analyzer.evaluate()
                
                # Extract structured data for session analysis
                from src.form_standards import assess_injury_risk, get_form_quality_score
                
                elbows = [f['elbow'] for f in analyzer.rep_angles if f['elbow']]
                hips = [f['hip'] for f in analyzer.rep_angles if f['hip']]
                
                analysis_angles = {
                    'left_elbow': min(elbows) if elbows else None,
                    'left_hip': min(hips) if hips else None,
                }
                
                bottom_landmarks = None
                if elbows:
                    min_elbow_idx = elbows.index(min(elbows))
                    bottom_landmarks = analyzer.rep_angles[min_elbow_idx]['landmarks']
                
                risk_assessment = assess_injury_risk(analysis_angles, bottom_landmarks)
                score = get_form_quality_score(risk_assessment)
                
                rep_data = {
                    'rep_num': reps,
                    'score': score,
                    'issues': risk_assessment.get('issues', []),
                    'flags': risk_assessment.get('flags', []),
                    'feedback': feedback
                }
                
                session.add_rep(rep_data)
                
                # ─── BRIEF PER-REP FEEDBACK ────────────────────────
                print(f"\n[REP {reps}] Score: {score}/100", flush=True)
                positives = [f for f in risk_assessment['feedback'] if "[GOOD]" in f]
                if positives:
                    print(f"  ✓ {positives[0].replace('[GOOD] ', '')}", flush=True)
                else:
                    print(f"  ✓ Form acceptable", flush=True)
                
                if risk_assessment['issues']:
                    for issue in risk_assessment['issues'][:2]:
                        print(f"  ⚠ {issue.replace('_', ' ').title()}", flush=True)
                
                last_feedback = [f"Rep {reps}: {score}/100"]

                analyzer.reset()

            # DRAW
            frame = estimator.draw_skeleton(frame, results)

            # UI
            cv2.putText(frame, f"Reps: {reps}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(frame, f"Stage: {stage}", (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if smooth_angle:
                cv2.putText(frame, f"Elbow: {smooth_angle:.1f}°", (20,110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            y = 140
            for msg in last_feedback:
                cv2.putText(frame, msg, (20,y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                y += 30

        cv2.imshow('FormAI - Real-time Push-up Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ─── FINALIZE IN-PROGRESS REP ────────────────────────
    if counter.state == "DOWN" and len(counter.angles_in_rep) > 10:
        if counter.update(None):  # Finalize video-end rep
            reps += 1
            analyzer.collect(None, None)  # Empty collect to mark completion
            try:
                feedback = analyzer.evaluate()
                
                from src.form_standards import assess_injury_risk, get_form_quality_score
                
                elbows = [f['elbow'] for f in analyzer.rep_angles if f['elbow']]
                hips = [f['hip'] for f in analyzer.rep_angles if f['hip']]
                
                analysis_angles = {
                    'left_elbow': min(elbows) if elbows else None,
                    'left_hip': min(hips) if hips else None,
                }
                
                bottom_landmarks = None
                if elbows:
                    min_elbow_idx = elbows.index(min(elbows))
                    bottom_landmarks = analyzer.rep_angles[min_elbow_idx]['landmarks']
                
                risk_assessment = assess_injury_risk(analysis_angles, bottom_landmarks)
                score = get_form_quality_score(risk_assessment)
                
                rep_data = {
                    'rep_num': reps,
                    'score': score,
                    'issues': risk_assessment.get('issues', []),
                    'flags': risk_assessment.get('flags', []),
                    'feedback': feedback
                }
                
                session.add_rep(rep_data)
                
            except:
                pass

    cap.release()
    cv2.destroyAllWindows()

    # ═══════════════════════════════════════════════════════════
    # GENERATE CONSOLIDATED SESSION REPORT
    # ═══════════════════════════════════════════════════════════
    session_report = session.analyze()
    
    print("\n")
    for line in session_report['feedback']:
        print(line, flush=True)
    
    print(f"\n{'='*60}")
    print(f"Total reps: {reps}")
    print(f"{'='*60}")
    
    # DEBUG: Show angle details
    if angle_samples:
        print(f"Angle range: {min(angle_samples):.1f}° - {max(angle_samples):.1f}°")
        print(f"Avg angle: {sum(angle_samples)/len(angle_samples):.1f}°")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "input.mp4"
    
    # Always output to same file
    output_file = "output.mp4"
    
    process_video(input_file, output_file)