"""
STANDARD PUSH-UP FORM - Angles and Posture Rules
Designed to prevent injury and ensure correct technique
"""

# ═══════════════════════════════════════════════════════════════════════
# JOINT ANGLE STANDARDS (in degrees)
# ═══════════════════════════════════════════════════════════════════════

class PushupStandards:
    """Define correct push-up form requirements"""
    
    # ELBOW ANGLE (most important for arm stress)
    ELBOW_ANGLE_OPTIMAL = 90  # Perfect at bottom
    ELBOW_ANGLE_GOOD_MIN = 75  # Prevent hyperextension
    ELBOW_ANGLE_GOOD_MAX = 95  # Safe range at bottom
    ELBOW_ANGLE_TOP = 160     # Extended position at top
    
    # HIP ANGLE (backbone of injury prevention)
    HIP_ANGLE_OPTIMAL = 175   # Neutral spine
    HIP_ANGLE_GOOD_MIN = 170  # Minimum to prevent sagging
    HIP_ANGLE_GOOD_MAX = 185  # Maximum before pike
    
    # KNEE ANGLE (prevent leg stress)
    KNEE_ANGLE_OPTIMAL = 180  # Full extension
    KNEE_ANGLE_GOOD_MIN = 175 # Nearly straight
    KNEE_ANGLE_GOOD_MAX = 180 # Full extension
    
    # SHOULDER ANGLE (prevent impingement)
    SHOULDER_WIDTH_OPTIMAL = 60   # From body (degrees)
    SHOULDER_WIDTH_MIN = 45       # Too narrow causes stress
    SHOULDER_WIDTH_MAX = 90       # Too wide causes impingement
    
    # WRIST ANGLE (prevent carpal tunnel)
    WRIST_ANGLE_OPTIMAL = 0   # Neutral
    WRIST_ANGLE_MIN = -30     # Some bending OK
    WRIST_ANGLE_MAX = 30      # Prevent excessive bend
    
    # HEAD POSITION (prevent neck strain)
    HEAD_ALIGNMENT_NEUTRAL = 0    # Aligned with spine
    HEAD_FORWARD_MAX = 15         # How much forward tilt is OK
    HEAD_DROP_MAX = 10            # How much down tilt is OK


# ═══════════════════════════════════════════════════════════════════════
# INJURY RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════

def assess_injury_risk(angles_dict, landmarks=None):
    """
    Analyze form and return injury risks
    
    Returns:
        {
            'risk_level': 'OK' | 'WARNING' | 'DANGER',
            'issues': [list of problems],
            'feedback': [list of corrections],
            'flags': [list of immediate concerns]
        }
    """
    issues = []
    feedback = []
    flags = []
    
    # Get all angles
    elbow = angles_dict.get('left_elbow') or angles_dict.get('right_elbow')
    hip = angles_dict.get('left_hip') or angles_dict.get('right_hip')
    knee = angles_dict.get('left_knee') or angles_dict.get('right_knee')
    
    # ─── ELBOW ASSESSMENT ────────────────────────────────────────
    if elbow:
        if elbow < PushupStandards.ELBOW_ANGLE_GOOD_MIN:
            flags.append("⚠️ ELBOW HYPEREXTENSION - Risk of joint damage")
            feedback.append("Don't lock elbows at bottom - stop at 75-80°")
            issues.append("hyperextended_elbow")
        elif elbow > PushupStandards.ELBOW_ANGLE_GOOD_MAX + 15:
            issues.append("insufficient_depth")
            feedback.append("Go deeper - elbows should bend more")
        elif PushupStandards.ELBOW_ANGLE_GOOD_MIN <= elbow <= PushupStandards.ELBOW_ANGLE_GOOD_MAX:
            feedback.append("✓ Elbow depth: Good (90° rule)")
        
    
    # ─── HIP ASSESSMENT ──────────────────────────────────────────
    if hip:
        if hip < PushupStandards.HIP_ANGLE_GOOD_MIN:
            flags.append("⚠️ SAGGING HIPS - Spine stress")
            feedback.append("Keep hips up - maintain straight line from head to heels")
            issues.append("sagging_hips")
        elif hip > PushupStandards.HIP_ANGLE_GOOD_MAX:
            flags.append("⚠️ PIKE POSITION - Shoulder overload")
            feedback.append("Lower hips - avoid pike position (butt in air)")
            issues.append("pike_position")
        elif PushupStandards.HIP_ANGLE_GOOD_MIN <= hip <= PushupStandards.HIP_ANGLE_GOOD_MAX:
            feedback.append("✓ Hip alignment: Good (neutral spine)")
    
    
    # ─── KNEE ASSESSMENT ────────────────────────────────────────
    if knee:
        if knee < PushupStandards.KNEE_ANGLE_GOOD_MIN:
            flags.append("⚠️ KNEE BENT - Not a full push-up")
            feedback.append("Straighten legs - knees should be locked (180°)")
            issues.append("bent_knees")
        else:
            feedback.append("✓ Leg extension: Good")
    
    
    # ─── DETERMINE RISK LEVEL ────────────────────────────────────
    if len(flags) > 0:
        risk_level = "DANGER" if any("hyperextension" in f or "sagging" in f for f in flags) else "WARNING"
    elif len(issues) > 0:
        risk_level = "WARNING"
    else:
        risk_level = "OK"
    
    
    return {
        'risk_level': risk_level,
        'issues': issues,
        'feedback': feedback,
        'flags': flags
    }


def get_form_quality_score(risk_assessment):
    """
    Convert risk assessment to a quality score (0-10)
    """
    score = 10
    
    # Deduct for each issue found
    score -= len(risk_assessment['issues']) * 0.5
    score -= len(risk_assessment['flags']) * 1.5
    
    return max(0, min(10, score))


def categorize_form(score):
    """Categorize form quality"""
    if score >= 9:
        return "EXCELLENT - Perfect form"
    elif score >= 7.5:
        return "GOOD - Minor adjustments needed"
    elif score >= 6:
        return "FAIR - Multiple form issues"
    else:
        return "POOR - Major form corrections needed"
