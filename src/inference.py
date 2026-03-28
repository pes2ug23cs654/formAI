import cv2
import pandas as pd

from src.angle_engine import LANDMARKS, get_all_angles, get_visibility
from src.form_analyzer import FormAnalyzer
from src.form_standards import assess_injury_risk, get_form_quality_score
from src.pose_estimator import PoseEstimator
from src.rep_counter import RepCounter
from src.squat_analyzer import SquatAnalyzer
from src.squat_rep_counter import SquatRepCounter
from src.squat_temporal_engine import SquatTemporalEngine
from src.temporal_engine import TemporalEngine


def _pick_elbow_angle(landmarks, angles):
    left_visible = get_visibility(landmarks, [
        LANDMARKS["left_shoulder"],
        LANDMARKS["left_elbow"],
        LANDMARKS["left_wrist"],
    ])

    right_visible = get_visibility(landmarks, [
        LANDMARKS["right_shoulder"],
        LANDMARKS["right_elbow"],
        LANDMARKS["right_wrist"],
    ])

    if left_visible:
        return angles.get("left_elbow")
    if right_visible:
        return angles.get("right_elbow")
    return None


def _pushup_analysis(cap, estimator):
    temporal = TemporalEngine()
    counter = RepCounter()
    analyzer = FormAnalyzer()

    reps = 0
    rep_scores = []
    issues = []
    angle_trace = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = estimator.process_frame(frame)
        landmarks = estimator.get_landmarks(results, frame.shape)
        if landmarks is None:
            continue

        angles = get_all_angles(landmarks)
        raw_angle = _pick_elbow_angle(landmarks, angles)
        smooth_angle = temporal.smooth(raw_angle)

        if smooth_angle is not None:
            angle_trace.append(smooth_angle)

        analyzer.collect(landmarks, angles)

        if counter.update(smooth_angle):
            reps += 1
            elbows = [f["elbow"] for f in analyzer.rep_angles if f["elbow"]]
            hips = [f["hip"] for f in analyzer.rep_angles if f["hip"]]

            analysis_angles = {
                "left_elbow": min(elbows) if elbows else None,
                "left_hip": min(hips) if hips else None,
            }

            risk = assess_injury_risk(analysis_angles, None)
            rep_scores.append(get_form_quality_score(risk))
            issues.extend(risk.get("issues", []))
            analyzer.reset()

    # Finalize partially completed rep at video end.
    if counter.update(None):
        reps += 1

    score = int(sum(rep_scores) / len(rep_scores)) if rep_scores else 75
    mistakes = sorted({i.replace("_", " ").title() for i in issues})
    if not mistakes:
        mistakes = ["Form looks good overall"]

    angles_df = pd.DataFrame({
        "frame": list(range(len(angle_trace))),
        "angle": angle_trace,
    })

    return {
        "score": score,
        "reps": reps,
        "mistakes": mistakes,
        "angles": angles_df,
    }


def _squat_analysis(cap, estimator):
    temporal = SquatTemporalEngine()
    counter = SquatRepCounter()
    analyzer = SquatAnalyzer()

    reps = 0
    rep_scores = []
    issues = []
    angle_trace = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = estimator.process_frame(frame)
        landmarks = estimator.get_landmarks(results, frame.shape)
        if landmarks is None:
            continue

        angles = get_all_angles(landmarks)
        knee_angle = angles.get("left_knee") or angles.get("right_knee")
        smooth_angle = temporal.smooth(knee_angle)

        if smooth_angle is not None:
            angle_trace.append(smooth_angle)

        if counter.update(smooth_angle, landmarks):
            reps += 1
            rep_angles = counter.get_last_rep()
            rep_landmarks = counter.get_last_landmarks()

            if rep_angles:
                result = analyzer.analyze(rep_angles, rep_landmarks)
                rep_scores.append(int(result.get("score", 75)))
                issues.extend(result.get("issues", []))

    if counter.update(None):
        reps += 1
        rep_angles = counter.get_last_rep()
        rep_landmarks = counter.get_last_landmarks()
        if rep_angles:
            result = analyzer.analyze(rep_angles, rep_landmarks)
            rep_scores.append(int(result.get("score", 75)))
            issues.extend(result.get("issues", []))

    score = int(sum(rep_scores) / len(rep_scores)) if rep_scores else 75
    mistakes = sorted({i.replace("_", " ").title() for i in issues})
    if not mistakes:
        mistakes = ["Form looks good overall"]

    angles_df = pd.DataFrame({
        "frame": list(range(len(angle_trace))),
        "angle": angle_trace,
    })

    return {
        "score": score,
        "reps": reps,
        "mistakes": mistakes,
        "angles": angles_df,
    }


def run_analysis(video_path, exercise):
    exercise = (exercise or "pushup").strip().lower()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    estimator = PoseEstimator()

    try:
        if exercise == "squat":
            return _squat_analysis(cap, estimator)
        return _pushup_analysis(cap, estimator)
    finally:
        cap.release()