import cv2
import numpy as np
import json
import sys

from src.pose_estimator import PoseEstimator
from src.angle_engine import get_all_angles, get_visibility, LANDMARKS
from src.temporal_engine import TemporalEngine
from src.rep_counter import RepCounter
from src.scorer import score_rep, summarize_session
from src.exercise_profiles import get_exercise_profile, build_analyzer
from src.exercise_rep_validation import ExerciseRepValidator, ExerciseSafetyValidator


DEFAULT_DESCENT_TRIGGER = 110.0
DEFAULT_ASCENT_THRESHOLD = 155.0

POSE_NOSE = 0
POSE_LEFT_HEEL = 29
POSE_RIGHT_HEEL = 30
POSE_LEFT_FOOT_INDEX = 31
POSE_RIGHT_FOOT_INDEX = 32


def _mean_visibility(landmarks, indices):
    values = [landmarks.get(i, {}).get("visibility", 0.0) for i in indices]
    return sum(values) / len(values) if values else 0.0


def get_elbow_with_confidence(landmarks, angles):
    left_indices = [
        LANDMARKS['left_shoulder'],
        LANDMARKS['left_elbow'],
        LANDMARKS['left_wrist'],
    ]
    right_indices = [
        LANDMARKS['right_shoulder'],
        LANDMARKS['right_elbow'],
        LANDMARKS['right_wrist'],
    ]

    left_visible = get_visibility(landmarks, left_indices)
    right_visible = get_visibility(landmarks, right_indices)

    left_conf = _mean_visibility(landmarks, left_indices)
    right_conf = _mean_visibility(landmarks, right_indices)

    if left_visible and (left_conf >= right_conf or not right_visible):
        return angles.get('left_elbow'), left_conf, "left"
    if right_visible:
        return angles.get('right_elbow'), right_conf, "right"

    # Fallback to higher-confidence side even if visibility gate is not fully met.
    if left_conf >= right_conf:
        return angles.get('left_elbow'), left_conf, "left"
    return angles.get('right_elbow'), right_conf, "right"


def get_knee_with_confidence(landmarks, angles):
    left_indices = [
        LANDMARKS['left_hip'],
        LANDMARKS['left_knee'],
        LANDMARKS['left_ankle'],
    ]
    right_indices = [
        LANDMARKS['right_hip'],
        LANDMARKS['right_knee'],
        LANDMARKS['right_ankle'],
    ]

    left_visible = get_visibility(landmarks, left_indices)
    right_visible = get_visibility(landmarks, right_indices)

    left_conf = _mean_visibility(landmarks, left_indices)
    right_conf = _mean_visibility(landmarks, right_indices)

    left_angle = angles.get('left_knee')
    right_angle = angles.get('right_knee')

    if left_visible and right_visible and left_angle is not None and right_angle is not None:
        return min(left_angle, right_angle), (left_conf + right_conf) / 2.0, "both"

    if left_visible and left_angle is not None:
        return left_angle, left_conf, "left"
    if right_visible and right_angle is not None:
        return right_angle, right_conf, "right"

    if left_angle is not None and right_angle is not None:
        return min(left_angle, right_angle), max(left_conf, right_conf), "both"
    if left_angle is not None:
        return left_angle, left_conf, "left"
    if right_angle is not None:
        return right_angle, right_conf, "right"
    return None, max(left_conf, right_conf), "none"


def get_hip_with_confidence(landmarks, angles):
    left_indices = [
        LANDMARKS['left_shoulder'],
        LANDMARKS['left_hip'],
        LANDMARKS['left_knee'],
    ]
    right_indices = [
        LANDMARKS['right_shoulder'],
        LANDMARKS['right_hip'],
        LANDMARKS['right_knee'],
    ]

    left_visible = get_visibility(landmarks, left_indices)
    right_visible = get_visibility(landmarks, right_indices)

    left_conf = _mean_visibility(landmarks, left_indices)
    right_conf = _mean_visibility(landmarks, right_indices)

    left_angle = angles.get('left_hip')
    right_angle = angles.get('right_hip')

    if left_visible and right_visible and left_angle is not None and right_angle is not None:
        return min(left_angle, right_angle), (left_conf + right_conf) / 2.0, "both"
    if left_visible and left_angle is not None:
        return left_angle, left_conf, "left"
    if right_visible and right_angle is not None:
        return right_angle, right_conf, "right"

    if left_angle is not None and right_angle is not None:
        return min(left_angle, right_angle), max(left_conf, right_conf), "both"
    if left_angle is not None:
        return left_angle, left_conf, "left"
    if right_angle is not None:
        return right_angle, right_conf, "right"
    return None, max(left_conf, right_conf), "none"


def get_primary_angle_with_confidence(profile, landmarks, angles):
    if profile.primary_angle == "elbow":
        return get_elbow_with_confidence(landmarks, angles)
    if profile.primary_angle == "knee":
        return get_knee_with_confidence(landmarks, angles)
    if profile.primary_angle == "hip":
        return get_hip_with_confidence(landmarks, angles)
    return None, 0.0, "none"


def _get_min_available_angle(angles, left_key, right_key):
    values = [angles.get(left_key), angles.get(right_key)]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return min(values)


def _mean_landmark_y(landmarks, indices, min_visibility=0.5):
    ys = []
    for idx in indices:
        lm = landmarks.get(idx)
        if not lm:
            continue
        if lm.get("visibility", 0.0) < min_visibility:
            continue
        ys.append(float(lm.get("y", 0.0)))
    if not ys:
        return None
    return sum(ys) / len(ys)


def _estimate_floor_y(landmarks, previous_floor_y=None):
    floor_candidates = []
    for idx in [
        LANDMARKS["left_ankle"],
        LANDMARKS["right_ankle"],
        POSE_LEFT_HEEL,
        POSE_RIGHT_HEEL,
        POSE_LEFT_FOOT_INDEX,
        POSE_RIGHT_FOOT_INDEX,
    ]:
        lm = landmarks.get(idx)
        if not lm:
            continue
        if lm.get("visibility", 0.0) < 0.45:
            continue
        floor_candidates.append(float(lm.get("y", 0.0)))

    if not floor_candidates:
        return previous_floor_y

    current_floor = max(floor_candidates)
    if previous_floor_y is None:
        return current_floor

    # Smooth floor estimate to reduce flicker from keypoint jitter.
    return (0.75 * previous_floor_y) + (0.25 * current_floor)


def _compute_pushup_floor_clearance(landmarks, floor_y):
    if landmarks is None or floor_y is None:
        return None

    left_hip = landmarks.get(LANDMARKS["left_hip"])
    right_hip = landmarks.get(LANDMARKS["right_hip"])
    left_ankle = landmarks.get(LANDMARKS["left_ankle"])
    right_ankle = landmarks.get(LANDMARKS["right_ankle"])

    leg_scales = []
    for hip, ankle in [(left_hip, left_ankle), (right_hip, right_ankle)]:
        if not hip or not ankle:
            continue
        if min(hip.get("visibility", 0.0), ankle.get("visibility", 0.0)) < 0.45:
            continue
        leg_scales.append(abs(float(ankle.get("y", 0.0)) - float(hip.get("y", 0.0))))

    if not leg_scales:
        return None

    body_scale = max(40.0, sum(leg_scales) / len(leg_scales))

    knee_y = _mean_landmark_y(
        landmarks,
        [LANDMARKS["left_knee"], LANDMARKS["right_knee"]],
        min_visibility=0.45,
    )
    chest_y = _mean_landmark_y(
        landmarks,
        [LANDMARKS["left_shoulder"], LANDMARKS["right_shoulder"], POSE_NOSE],
        min_visibility=0.45,
    )

    if knee_y is None or chest_y is None:
        return None

    knee_clearance_px = max(0.0, floor_y - knee_y)
    chest_clearance_px = max(0.0, floor_y - chest_y)

    knee_min_clearance_px = 0.11 * body_scale
    chest_min_clearance_px = 0.08 * body_scale

    return {
        "floor_y": floor_y,
        "body_scale": body_scale,
        "knee_clearance_px": knee_clearance_px,
        "chest_clearance_px": chest_clearance_px,
        "knee_min_clearance_px": knee_min_clearance_px,
        "chest_min_clearance_px": chest_min_clearance_px,
    }


def is_pushup_ready_for_count(angles, floor_clearance=None):
    """Gate push-up counting to avoid kneeling/seal-style false reps."""
    if not angles:
        return False

    hip_angle = _get_min_available_angle(angles, "left_hip", "right_hip")
    knee_angle = _get_min_available_angle(angles, "left_knee", "right_knee")

    if hip_angle is None or knee_angle is None:
        return False

    # Require near-plank posture and reasonably straight legs.
    angle_ready = hip_angle >= 155 and knee_angle >= 165
    if not angle_ready:
        return False

    # Missing floor reference is common in mobile videos; do not hard-block counts.
    if floor_clearance is None:
        return True

    # Keep this filter lenient so fast reps with slight jitter are still tracked.
    if floor_clearance["knee_clearance_px"] < (0.75 * floor_clearance["knee_min_clearance_px"]):
        return False
    if floor_clearance["chest_clearance_px"] < (0.70 * floor_clearance["chest_min_clearance_px"]):
        return False

    return True


def _calibrate_thresholds(calibration_angles, profile):
    """Derive thresholds from observed range while keeping safe bounds."""
    if len(calibration_angles) < 8:
        return {
            "used": False,
            "reason": "not_enough_samples",
            "sample_count": len(calibration_angles),
            "descent_trigger": profile.default_descent_trigger,
            "ascent_threshold": profile.default_ascent_threshold,
            "angle_min": None,
            "angle_max": None,
        }

    angle_min = float(np.percentile(calibration_angles, 10))
    angle_max = float(np.percentile(calibration_angles, 90))

    dynamic_descent = angle_min + 0.45 * (angle_max - angle_min)
    dynamic_ascent = angle_min + 0.85 * (angle_max - angle_min)

    descent_trigger = float(np.clip(dynamic_descent, profile.calibration_descent_bounds[0], profile.calibration_descent_bounds[1]))
    ascent_threshold = float(np.clip(dynamic_ascent, profile.calibration_ascent_bounds[0], profile.calibration_ascent_bounds[1]))

    return {
        "used": True,
        "reason": "calibrated",
        "sample_count": len(calibration_angles),
        "descent_trigger": round(descent_trigger, 1),
        "ascent_threshold": round(ascent_threshold, 1),
        "angle_min": round(angle_min, 1),
        "angle_max": round(angle_max, 1),
    }


def process_video(
    video_path,
    output_path,
    report_json_path=None,
    debug=True,
    calibration_seconds=3,
    confidence_threshold=0.55,
    exercise="pushup",
):

    profile = get_exercise_profile(exercise)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames_estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    estimator = PoseEstimator()
    temporal = TemporalEngine(buffer_size=2 if profile.key == "pushup" else 3)
    counter = RepCounter(
        descent_trigger=profile.default_descent_trigger,
        ascent_threshold=profile.default_ascent_threshold,
        require_initial_lockout=(profile.key == "pushup"),
        initial_lockout_frames=max(2, int(round(0.06 * fps))),
    )
    analyzer = build_analyzer(profile.key)

    reps = 0
    last_feedback = []
    rep_reports = []
    
    frame_count = 0
    angle_samples = []
    low_confidence_frames = 0
    used_for_rep_logic_frames = 0
    readiness_rejected_frames = 0
    floor_clearance_rejected_frames = 0
    pushup_ready_streak = 0
    pushup_not_ready_streak = 0
    read_failure_streak = 0
    max_read_failures = 10
    floor_y_estimate = None
    posture_hint_counts = {
        "knees_too_low": 0,
        "chest_too_low": 0,
        "missing_floor_reference": 0,
    }

    # Calibration fallback: if video is short, use up to half its frames (min 15 frames).
    desired_calibration_frames = int(max(1, calibration_seconds) * fps)
    short_video_mode = total_frames_estimate > 0 and total_frames_estimate < desired_calibration_frames
    if total_frames_estimate > 0:
        calibration_frames = min(desired_calibration_frames, max(15, total_frames_estimate // 2))
    else:
        calibration_frames = desired_calibration_frames
    calibration_angles = []
    calibration_applied = False
    calibration_result = {
        "used": False,
        "reason": "pending",
        "sample_count": 0,
        "descent_trigger": profile.default_descent_trigger,
        "ascent_threshold": profile.default_ascent_threshold,
        "angle_min": None,
        "angle_max": None,
        "target_frames": calibration_frames,
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            read_failure_streak += 1
            # Recover from occasional decode failures in high-motion mobile videos.
            if read_failure_streak <= max_read_failures:
                continue
            break

        read_failure_streak = 0

        frame_count += 1

        results = estimator.process_frame(frame)
        landmarks = estimator.get_landmarks(results, frame.shape)
        angles = get_all_angles(landmarks)

        if landmarks is None:
            out.write(frame)
            continue

        primary_angle, primary_confidence, side_used = get_primary_angle_with_confidence(profile, landmarks, angles)

        if primary_angle is not None and frame_count <= calibration_frames:
            calibration_angles.append(primary_angle)

        if not calibration_applied and frame_count >= calibration_frames:
            calibration_result = _calibrate_thresholds(calibration_angles, profile)
            if short_video_mode and calibration_result["used"]:
                calibration_result["reason"] = "short_video_calibrated"
            counter.set_thresholds(
                descent_trigger=calibration_result["descent_trigger"],
                ascent_threshold=calibration_result["ascent_threshold"],
            )
            calibration_applied = True
            if debug:
                print(
                    "[CALIBRATION] "
                    f"used={calibration_result['used']} "
                    f"samples={calibration_result['sample_count']} "
                    f"descent={calibration_result['descent_trigger']} "
                    f"ascent={calibration_result['ascent_threshold']}",
                    flush=True,
                )

        analysis_confidence_ok = primary_angle is not None and primary_confidence >= confidence_threshold
        count_confidence_threshold = confidence_threshold
        if profile.key == "pushup":
            # Fast reps can momentarily reduce landmark confidence; allow a small
            # buffer for counting transitions while keeping strict analysis gating.
            count_confidence_threshold = max(0.40, confidence_threshold - 0.10)

        confidence_ok = primary_angle is not None and primary_confidence >= count_confidence_threshold
        if not confidence_ok:
            low_confidence_frames += 1

        # TRACK MOTION
        smooth_angle = temporal.smooth(primary_angle) if confidence_ok else temporal.last_valid
        stage = temporal.detect_stage(smooth_angle)
        
        # DEBUG: Sample angles every 50 frames (more frequent)
        if debug and frame_count % 50 == 0 and smooth_angle:
            angle_samples.append(smooth_angle)
            print(
                f"[FRAME {frame_count:4d}] Angle: {smooth_angle:6.1f}° | "
                f"Conf: {primary_confidence:.2f} | Side: {side_used} | State: {counter.state}",
                flush=True,
            )

        # Collect analysis only on reliable frames to reduce false warnings.
        if analysis_confidence_ok:
            used_for_rep_logic_frames += 1
            analyzer.collect(landmarks, angles)

        count_ready = True
        current_posture_hints = []
        
        # ─── EXERCISE-SPECIFIC READINESS CHECKS ──────────────────
        if profile.key == "pushup":
            floor_y_estimate = _estimate_floor_y(landmarks, floor_y_estimate)
            floor_clearance = _compute_pushup_floor_clearance(landmarks, floor_y_estimate)
            posture_ready = is_pushup_ready_for_count(angles, floor_clearance=floor_clearance)

            floor_rejected = False
            if floor_clearance is None:
                floor_rejected = True
                posture_hint_counts["missing_floor_reference"] += 1
                current_posture_hints.append("Need full side view (feet + torso visible)")
            else:
                if floor_clearance["knee_clearance_px"] < floor_clearance["knee_min_clearance_px"]:
                    floor_rejected = True
                    posture_hint_counts["knees_too_low"] += 1
                    current_posture_hints.append("Lift knees off floor; legs straighter")
                if floor_clearance["chest_clearance_px"] < floor_clearance["chest_min_clearance_px"]:
                    floor_rejected = True
                    posture_hint_counts["chest_too_low"] += 1
                    current_posture_hints.append("Keep chest off ground between reps")

            if floor_rejected:
                floor_clearance_rejected_frames += 1

            if posture_ready:
                pushup_ready_streak += 1
                pushup_not_ready_streak = 0
            else:
                pushup_ready_streak = 0
                pushup_not_ready_streak += 1
                readiness_rejected_frames += 1

            # For push-ups, always allow counting transitions and classify quality later.
            # This avoids missing fast reps due transient posture/readiness gating.
            count_ready = True
            if current_posture_hints:
                last_feedback = current_posture_hints[:2]

        # COUNT REPS only on reliable frames
        if confidence_ok and count_ready and counter.update(smooth_angle):

            completed_angles = counter.get_last_completed_rep_angles() or []

            # Ignore startup pseudo-rep from initial positioning in fast videos.
            if profile.key == "pushup" and reps == 0 and completed_angles:
                startup_frames_guard = max(30, int(round(1.0 * fps)))
                if frame_count <= startup_frames_guard:
                    if debug:
                        print(
                            f"[REP SKIPPED] startup window (frame={frame_count}, guard={startup_frames_guard})",
                            flush=True,
                        )
                    analyzer.reset()
                    continue

            # For curls, reject half reps (short ROM) before scoring as full reps.
            if profile.key == "bicep_curl" and completed_angles:
                rep_min = min(completed_angles)
                rep_max = max(completed_angles)
                rep_span = rep_max - rep_min

                has_bottom_extension = rep_max >= 145.0
                has_top_contraction = rep_min <= 105.0
                has_enough_rom = rep_span >= 45.0

                if not (has_bottom_extension and has_top_contraction and has_enough_rom):
                    if debug:
                        print(
                            "[REP SKIPPED] partial bicep curl "
                            f"(min={rep_min:.1f}, max={rep_max:.1f}, span={rep_span:.1f})",
                            flush=True,
                        )
                    analyzer.reset()
                    continue

            last_feedback = analyzer.evaluate()
            rep_score = score_rep(last_feedback)

            # ─── EXERCISE-SPECIFIC REP VALIDATION ──────────────────────
            should_count_rep, validation_reason = ExerciseRepValidator.validate_rep(
                profile.key, rep_score, last_feedback
            )

            # Count every detected movement rep attempt, and track validity separately.
            reps += 1
            if not should_count_rep:
                rep_score["is_valid"] = False
                invalid_reasons = set(rep_score.get("invalid_reasons", []))
                invalid_reasons.add("rep_rejected")
                rep_score["invalid_reasons"] = sorted(invalid_reasons)

            rep_reports.append({
                "rep_number": reps,
                "feedback": last_feedback,
                "counted_valid": bool(should_count_rep),
                "validation_reason": validation_reason,
                **rep_score,
            })

            if (not should_count_rep) and debug:
                print(f"[REP FLAGGED INVALID] {profile.key}: {validation_reason}", flush=True)

            if debug:
                shown_rep_number = reps if should_count_rep else reps + 1
                print(f"\n*** REP {shown_rep_number} ***", flush=True)
                for msg in last_feedback:
                    print(f"  - {msg}", flush=True)
                print(f"  - Score: {rep_score['score']}", flush=True)
                print(f"  - Valid: {rep_score['is_valid']}", flush=True)
                print(f"  - Counted: {should_count_rep}", flush=True)

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
            completed_angles = counter.get_last_completed_rep_angles() or []

            if profile.key == "bicep_curl" and completed_angles:
                rep_min = min(completed_angles)
                rep_max = max(completed_angles)
                rep_span = rep_max - rep_min

                has_bottom_extension = rep_max >= 145.0
                has_top_contraction = rep_min <= 105.0
                has_enough_rom = rep_span >= 45.0

                if not (has_bottom_extension and has_top_contraction and has_enough_rom):
                    if debug:
                        print(
                            "[REP SKIPPED] final partial bicep curl "
                            f"(min={rep_min:.1f}, max={rep_max:.1f}, span={rep_span:.1f})",
                            flush=True,
                        )
                    analyzer.reset()
                    completed_angles = []
                
            if profile.key == "bicep_curl" and not completed_angles:
                pass
            else:
                analyzer.collect(None, None)  # Empty collect to mark completion
                try:
                    feedback = analyzer.evaluate()
                    last_feedback = feedback[:3]  # Show last 3 lines
                    rep_score = score_rep(feedback)
                    
                    # ─── EXERCISE-SPECIFIC REP VALIDATION ──────────────────────
                    should_count_rep, validation_reason = ExerciseRepValidator.validate_rep(
                        profile.key, rep_score, feedback
                    )

                    reps += 1
                    if not should_count_rep:
                        rep_score["is_valid"] = False
                        invalid_reasons = set(rep_score.get("invalid_reasons", []))
                        invalid_reasons.add("rep_rejected")
                        rep_score["invalid_reasons"] = sorted(invalid_reasons)

                    rep_reports.append({
                        "rep_number": reps,
                        "feedback": feedback,
                        "counted_valid": bool(should_count_rep),
                        "validation_reason": validation_reason,
                        **rep_score,
                    })

                    if debug:
                        print(f"\n{'='*50}")
                        print(f"Rep #{reps} (FINAL):")
                        for msg in feedback:
                            print(msg)
                        print(f"Score: {rep_score['score']} | Valid: {rep_score['is_valid']}")
                        print(f"Counted: {should_count_rep}")
                        print(f"{'='*50}")
                except Exception:
                    last_feedback = [f"Rep #{reps} - Analysis pending"]

    cap.release()
    out.release()

    # Fallback apply: very short clips may end before target calibration frame.
    if not calibration_applied:
        calibration_result = _calibrate_thresholds(calibration_angles, profile)
        calibration_result["reason"] = (
            "short_video_fallback"
            if calibration_result["used"]
            else "short_video_default_thresholds"
        )
        counter.set_thresholds(
            descent_trigger=calibration_result["descent_trigger"],
            ascent_threshold=calibration_result["ascent_threshold"],
        )
        calibration_applied = True

    session_summary = summarize_session(rep_reports)

    session_coaching_tips = []
    
    # Exercise-specific coaching tips
    if profile.key == "pushup":
        if floor_clearance_rejected_frames > 0:
            session_coaching_tips.append("Push-up setup rejected often: keep knees and chest off floor in plank position.")
        if posture_hint_counts["knees_too_low"] > 0:
            session_coaching_tips.append("Straighten legs more; avoid knee-supported reps.")
        if posture_hint_counts["chest_too_low"] > 0:
            session_coaching_tips.append("Do not rest chest on ground between reps.")
        if posture_hint_counts["missing_floor_reference"] > 0:
            session_coaching_tips.append("Record from a clean side angle with full body visible.")
        if reps == 0 and readiness_rejected_frames > 0:
            session_coaching_tips.append("No reps counted because push-up readiness conditions were not met consistently.")
    
    elif profile.key == "squat":
        if reps > 0:
            avg_score = session_summary.get("avg_score", 0)
            if avg_score < 70:
                session_coaching_tips.append("Work on squat depth - aim for at least parallel (hips level with knees).")
        if readiness_rejected_frames > 0 * 0.3:
            session_coaching_tips.append("Ensure you're starting from a standing position between reps.")
    
    elif profile.key == "lunge":
        if reps > 0:
            avg_score = session_summary.get("avg_score", 0)
            if avg_score < 70:
                session_coaching_tips.append("Maintain better balance - front knee should align over ankle.")
        session_coaching_tips.append("Keep torso upright with minimal forward lean.")
    
    elif profile.key == "bicep_curl":
        if reps > 0:
            avg_score = session_summary.get("avg_score", 0)
            if avg_score < 70:
                session_coaching_tips.append("Control the weight - avoid using momentum (jerky motions).")
        session_coaching_tips.append("Fully extend and contract arms for complete range of motion.")
    
    elif profile.key == "shoulder_press":
        if reps > 0:
            avg_score = session_summary.get("avg_score", 0)
            if avg_score < 70:
                session_coaching_tips.append("Lock out fully at the top - extend elbows completely overhead.")
        session_coaching_tips.append("Maintain core stability - avoid excessive lower back arching.")
    
    elif profile.key == "situp":
        if reps > 0:
            avg_score = session_summary.get("avg_score", 0)
            if avg_score < 70:
                session_coaching_tips.append("Use your core, not your neck - don't pull yourself up by your head.")
        session_coaching_tips.append("Go through full range of motion - lie back completely and sit up fully.")
    
    elif profile.key == "mountain_climber":
        if reps > 0:
            avg_score = session_summary.get("avg_score", 0)
            if avg_score < 70:
                session_coaching_tips.append("Keep hips level in plank - no sagging or piking.")
        session_coaching_tips.append("Drive knees towards chest with controlled, rhythmic motion.")

    if session_coaching_tips:
        session_summary["coaching_tips"] = list(dict.fromkeys(session_coaching_tips))

    if reps == 0 and not session_summary.get("top_issues") and session_coaching_tips:
        session_summary["top_issues"] = [f"{profile.key}_setup_issues"]

    report = {
        "input_video": video_path,
        "output_video": output_path,
        "exercise": profile.key,
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": frame_count,
            "estimated_total_frames": total_frames_estimate,
        },
        "calibration": calibration_result,
        "confidence": {
            "threshold": confidence_threshold,
            "low_confidence_frames": low_confidence_frames,
            "used_for_rep_logic_frames": used_for_rep_logic_frames,
            "readiness_rejected_frames": readiness_rejected_frames,
            "floor_clearance_rejected_frames": floor_clearance_rejected_frames,
            "posture_hint_counts": posture_hint_counts,
        },
        "summary": {
            "total_reps": reps,
            **session_summary,
        },
        "rep_reports": rep_reports,
        "angle_stats": {
            "sampled_count": len(angle_samples),
            "sampled_min": round(min(angle_samples), 1) if angle_samples else None,
            "sampled_max": round(max(angle_samples), 1) if angle_samples else None,
            "sampled_avg": round(sum(angle_samples) / len(angle_samples), 1) if angle_samples else None,
        },
    }

    if report_json_path:
        with open(report_json_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)

    if debug:
        print(f"\n{'='*50}")
        print(f"Total reps: {reps}")
        print(f"Valid reps: {session_summary['valid_reps']} | Invalid reps: {session_summary['invalid_reps']}")
        print(f"Average score: {session_summary['avg_score']}")
        print(f"{'='*50}")

        if angle_samples:
            print(f"Angle range: {min(angle_samples):.1f}° - {max(angle_samples):.1f}°")
            print(f"Avg angle: {sum(angle_samples)/len(angle_samples):.1f}°")

    return report

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "input/input.mp4"
    
    # Always output to same file
    output_file = "output/output.mp4"
    
    process_video(input_file, output_file)