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
    ELBOW_ANGLE_GOOD_MIN = 75  # Prevent hyperextension at bottom
    ELBOW_ANGLE_GOOD_MAX = 95  # Safe range at bottom
    ELBOW_ANGLE_TOP = 160     # Extended position at top
    ELBOW_ANGLE_MAX_SAFE = 175  # Do NOT exceed this at top (hyperextension risk)
    ELBOW_ANGLE_TOO_DEEP = 60   # Below this = excessive depth = shoulder stress
    
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
    
    # HEAD POSITION (prevent neck strain) - ratio-based
    HEAD_ALIGNMENT_NEUTRAL = 0    # Aligned with spine
    HEAD_DROP_MAX_RATIO = 0.25    # Max drop as ratio of torso length (camera-independent)
    HEAD_FORWARD_MAX = 15         # How much forward tilt is OK
    HEAD_DROP_MAX = 10            # How much down tilt is OK


# ═══════════════════════════════════════════════════════════════════════
# INJURY RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════

def assess_injury_risk(angles_dict, landmarks=None):
    """
    Analyze form and return injury risks with severity levels
    
    Returns:
        {
            'risk_level': 'OK' | 'WARNING' | 'DANGER',
            'severity': 'minor' | 'moderate' | 'critical',
            'issues': [list of problems],
            'feedback': [list of corrections],
            'flags': [list of immediate concerns]
        }
    """
    issues = []
    feedback = []
    flags = []
    severity_issues = {'minor': [], 'moderate': [], 'critical': []}
    
    # Get all angles
    elbow = angles_dict.get('left_elbow') or angles_dict.get('right_elbow')
    hip = angles_dict.get('left_hip') or angles_dict.get('right_hip')
    knee = angles_dict.get('left_knee') or angles_dict.get('right_knee')
    
    # ─── ELBOW ASSESSMENT ────────────────────────────────────────
    if elbow:
        # CRITICAL: Too deep (excessive shoulder stress)
        if elbow < PushupStandards.ELBOW_ANGLE_TOO_DEEP:  # < 60°
            flags.append(f"[CRITICAL] EXCESSIVE DEPTH {elbow:.0f}° - shoulder stress risk")
            feedback.append(f"Don't go so deep - stop at 75-80° to protect shoulders")
            issues.append("excessive_depth")
            severity_issues['critical'].append("excessive_depth")
        # CRITICAL: Hyperextension at top
        elif elbow > PushupStandards.ELBOW_ANGLE_MAX_SAFE:  # > 175°
            flags.append(f"[CRITICAL] HYPEREXTENSION {elbow:.0f}° - joint damage risk")
            feedback.append(f"Don't hyperextend elbows - keep below 175°")
            issues.append("hyperextended_elbow")
            severity_issues['critical'].append("hyperextension")
        # CRITICAL: Too shallow during rep
        elif elbow > PushupStandards.ELBOW_ANGLE_GOOD_MAX + 15:  # > 110° at bottom
            issues.append("insufficient_depth")
            feedback.append("Go deeper - elbows should bend to 75-95°")
            severity_issues['moderate'].append("insufficient_depth")
        # MODERATE: Slightly shallow
        elif elbow > PushupStandards.ELBOW_ANGLE_GOOD_MAX:  # 95-110°
            severity_issues['minor'].append("slightly_shallow")
            feedback.append(f"Slightly shallow ({elbow:.0f}°) - try for 75-95°")
        elif PushupStandards.ELBOW_ANGLE_GOOD_MIN <= elbow <= PushupStandards.ELBOW_ANGLE_GOOD_MAX:
            feedback.append("[GOOD] Elbow depth: Excellent (75-95° range)")
    
    # ─── HIP ASSESSMENT ──────────────────────────────────────────
    if hip:
        if hip < 165:  # Severe sagging
            flags.append(f"[CRITICAL] SEVERE SAGGING {hip:.0f}° - spine damage risk")
            feedback.append("Keep hips up - maintain straight line from head to heels")
            issues.append("sagging_hips")
            severity_issues['critical'].append("sagging_hips")
        elif hip < PushupStandards.HIP_ANGLE_GOOD_MIN:  # 165-170°
            flags.append(f"[MODERATE] Sagging hips {hip:.0f}° - spine stress")
            feedback.append("Hips sagging - tighten core and keep body straight")
            issues.append("mild_sagging")
            severity_issues['moderate'].append("sagging_hips")
        elif hip > PushupStandards.HIP_ANGLE_GOOD_MAX:  # Pike position
            flags.append("[MODERATE] PIKE POSITION - shoulder overload")
            feedback.append("Lower hips - avoid pike position (butt in air)")
            issues.append("pike_position")
            severity_issues['moderate'].append("pike_position")
        elif PushupStandards.HIP_ANGLE_GOOD_MIN <= hip <= PushupStandards.HIP_ANGLE_GOOD_MAX:
            feedback.append("[GOOD] Hip alignment: Perfect (neutral spine)")
    
    # ─── KNEE ASSESSMENT ────────────────────────────────────────
    if knee:
        if knee < PushupStandards.KNEE_ANGLE_GOOD_MIN:
            flags.append("[FLAG] KNEE BENT - Not a full push-up")
            feedback.append("Straighten legs - knees should be locked (180°)")
            issues.append("bent_knees")
            severity_issues['moderate'].append("bent_knees")
        else:
            feedback.append("[GOOD] Leg extension: Perfect")
    
    # ─── DETERMINE RISK & SEVERITY LEVEL ────────────────────────
    if severity_issues['critical']:
        risk_level = "DANGER"
        severity = "critical"
    elif severity_issues['moderate'] or len(flags) > 0:
        risk_level = "WARNING"
        severity = "moderate"
    elif severity_issues['minor'] or len(issues) > 0:
        risk_level = "WARNING"
        severity = "minor"
    else:
        risk_level = "OK"
        severity = "none"
    
    return {
        'risk_level': risk_level,
        'severity': severity,
        'issues': issues,
        'feedback': feedback,
        'flags': flags,
        'severity_breakdown': severity_issues
    }


def get_form_quality_score(risk_assessment):
    """
    Convert risk assessment to a quality score (0-100) with severity weighting
    """
    score = 100
    
    # Deduct heavily for critical issues
    if risk_assessment.get('severity_breakdown', {}).get('critical'):
        score -= len(risk_assessment['severity_breakdown']['critical']) * 20
    
    # Deduct moderately for moderate issues
    if risk_assessment.get('severity_breakdown', {}).get('moderate'):
        score -= len(risk_assessment['severity_breakdown']['moderate']) * 10
    
    # Deduct slightly for minor issues
    if risk_assessment.get('severity_breakdown', {}).get('minor'):
        score -= len(risk_assessment['severity_breakdown']['minor']) * 3
    
    return max(0, min(100, score))


def categorize_form(score):
    """Categorize form quality"""
    if score >= 90:
        return "EXCELLENT - Perfect form"
    elif score >= 75:
        return "GOOD - Minor adjustments needed"
    elif score >= 60:
        return "FAIR - Multiple form issues"
    else:
        return "POOR - Major form corrections needed"
