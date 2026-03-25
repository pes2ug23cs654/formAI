def evaluate_pushup(min_elbow, max_elbow, body_angles):
    """
    Evaluate push-up form quality and provide feedback
    Enhanced with false positive detection
    """
    feedback = []
    avg_body = sum(body_angles) / len(body_angles) if body_angles else 0

    # DEPTH CHECK (calibrated from input.mp4 analysis)
    # Min observed: 51.6° | Now adjusted for long.mp4: 83.7° min
    # Using range: 70-95° for "good depth"
    if min_elbow < 70:
        feedback.append("Too deep (joint stress)")
    elif min_elbow > 95:
        feedback.append("Not deep enough")
    else:
        feedback.append("Good depth")

    # BODY ALIGNMENT CHECK (consistent with temporal_engine.detect_stage)
    if 160 <= avg_body <= 195:
        feedback.append("Good body alignment")
        body_ok = True
    else:
        feedback.append("Bad body alignment")
        body_ok = False

    return feedback, body_ok


def is_false_positive(min_elbow, max_elbow, angle_range, rep_frame_count=None):
    """
    Detect if rep is a false positive (kneeling, arm wave, idle motion)
    
    Args:
        min_elbow: Minimum elbow angle during cycle
        max_elbow: Maximum elbow angle during cycle
        angle_range: max_elbow - min_elbow
        rep_frame_count: Number of frames in the rep (for speed estimation)
        
    Returns:
        is_false: Bool - True if looks like false positive
    """
    
    # KNEELING DETECTION
    # Kneeling: arms extended but never go deep (stays >70°)
    if min_elbow > 70 and max_elbow > 140:
        return True
    
    # ARM WAVE DETECTION  
    # Arm wave: extends top but barely descends, no depth
    if max_elbow > 150 and min_elbow > 120:
        return True
    
    # INSUFFICIENT RANGE
    # Real pushup has 60°+ range, less indicates incomplete motion
    if angle_range < 60:
        return True
    
    # VERY SLOW REP (likely pose tracking drift)
    if rep_frame_count and rep_frame_count > 200:
        if angle_range < 70:
            return True
    
    return False