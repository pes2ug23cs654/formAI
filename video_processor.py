import cv2
import numpy as np
import sys

from src.pose_estimator import PoseEstimator
from src.angle_engine import get_all_angles, get_visibility, LANDMARKS
from src.temporal_engine import TemporalEngine
from src.rep_counter import RepCounter
from src.rep_analyzer import RepAnalyzer
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
    analyzer = RepAnalyzer()

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
            print(f"[FRAME {frame_count:4d}] Angle: {smooth_angle:6.1f}° | State: {counter.state}", flush=True)

        # ALWAYS COLLECT
        analyzer.collect(landmarks, angles)

        # COUNT REPS (assume already in plank)
        if counter.update(smooth_angle):
            reps += 1

            last_feedback = analyzer.evaluate()

            print(f"\n*** REP {reps} ***", flush=True)
            for msg in last_feedback:
                print(f"  - {msg}", flush=True)

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
                last_feedback = feedback[:3]  # Show last 3 lines
                print(f"\n{'─'*50}")
                print(f"Rep #{reps} (FINAL):")
                for msg in feedback:
                    print(msg)
                print(f"{'─'*50}")
            except:
                last_feedback = [f"Rep #{reps} - Analysis pending"]

    cap.release()
    out.release()

    print(f"\n{'='*50}")
    print(f"Total reps: {reps}")
    print(f"{'='*50}")
    
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