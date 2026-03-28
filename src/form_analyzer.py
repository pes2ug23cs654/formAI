"""
Enhanced Form Analyzer - Comprehensive per-rep form quality assessment
Focuses on injury prevention and proper technique
"""

import math
import numpy as np

from src.form_standards import (
    assess_injury_risk,
    get_form_quality_score,
    categorize_form
)


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points (3D-aware, camera-independent)"""
    dx = p1.get('x', 0) - p2.get('x', 0)
    dy = p1.get('y', 0) - p2.get('y', 0)
    dz = (p1.get('z', 0) - p2.get('z', 0)) * 100  # normalize z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


class FormAnalyzer:
    """Track and analyze form for every repetition"""

    def __init__(self):
        self.rep_angles = []
        self.current_rep = 0
        self.head_readings = []
        self.head_baseline = None
        self.calibration_complete = False

    def collect(self, landmarks, angles):
        """Collect frame data + calibrate head position"""

        # ===== HEAD CALIBRATION =====
        if not self.calibration_complete and landmarks and angles:

            elbow = angles.get('left_elbow') or angles.get('right_elbow')
            hip_angle = angles.get('left_hip') or angles.get('right_hip')

            # Only calibrate in TOP position
            if elbow and hip_angle and elbow > 160 and hip_angle > 165:
                try:
                    head = landmarks[0]
                    shoulder = landmarks[11]
                    hip = landmarks[23]

                    body_length = euclidean_distance(shoulder, hip)
                    if body_length < 10:
                        return

                    head_drop = euclidean_distance(head, shoulder)
                    ratio = head_drop / body_length

                    # Filter bad posture
                    if 0.15 < ratio < 0.45:
                        self.head_readings.append(ratio)

                    if len(self.head_readings) >= 20:
                        self.head_baseline = np.median(self.head_readings)
                        self.calibration_complete = True
                        print(f"[HEAD CALIBRATED] Baseline ratio: {self.head_baseline:.3f}", flush=True)

                except Exception:
                    pass

        # ===== NORMAL DATA COLLECTION =====
        if angles:
            elbow = angles.get('left_elbow') or angles.get('right_elbow')
            hip = angles.get('left_hip') or angles.get('right_hip')
            knee = angles.get('left_knee') or angles.get('right_knee')

            if elbow:
                head_y = landmarks[0].get('y') if landmarks else None

                self.rep_angles.append({
                    'elbow': elbow,
                    'hip': hip,
                    'knee': knee,
                    'landmarks': landmarks,
                    'head_y': head_y
                })

    def _check_head_position(self, bottom_landmarks, elbow_angle=None):
        """Detect head dropping relative to calibrated baseline"""

        if not bottom_landmarks or len(bottom_landmarks) < 24:
            return None

        try:
            head = bottom_landmarks[0]
            shoulder = bottom_landmarks[11]
            hip = bottom_landmarks[23]

            body_length = euclidean_distance(shoulder, hip)
            if body_length < 10:
                return None

            head_drop = euclidean_distance(head, shoulder)
            ratio = head_drop / body_length

            print(f"[HEAD] Ratio: {ratio:.3f}", flush=True)

            # Only evaluate at bottom position
            if elbow_angle and elbow_angle > 120:
                return None

            # ===== CALIBRATED =====
            if self.head_baseline is not None:
                deviation = ratio - self.head_baseline
                print(f"[HEAD DEBUG] Dev: {deviation:.3f}", flush=True)

                if deviation > 0.15:
                    return "[HEAD] Dropping excessively - keep neck neutral"
                elif deviation > 0.08:
                    return "[HEAD] Head slightly low - improve neck alignment"

                return None

            # ===== FALLBACK =====
            else:
                # ===== FALLBACK (TUNED 🔥) =====
                if ratio > 0.45:
                     return "[HEAD] Dropping excessively - keep neck neutral"
                elif ratio > 0.38:
                    return "[HEAD] Head slightly low - improve neck alignment"

                return None

        except Exception:
            return None

    def evaluate(self):
        """Evaluate one full repetition"""

        if not self.rep_angles:
            return ["No data collected"]

        self.current_rep += 1

        elbows = [f['elbow'] for f in self.rep_angles if f['elbow']]
        hips = [f['hip'] for f in self.rep_angles if f['hip']]

        analysis_angles = {
            'left_elbow': min(elbows) if elbows else None,
            'right_elbow': min(elbows) if elbows else None,
            'left_hip': min(hips) if hips else None,
            'right_hip': min(hips) if hips else None,
        }

        bottom_landmarks = None
        min_elbow = None

        if elbows:
            min_elbow = min(elbows)
            min_idx = elbows.index(min_elbow)
            bottom_landmarks = self.rep_angles[min_idx]['landmarks']

        risk_assessment = assess_injury_risk(analysis_angles, bottom_landmarks)
        quality_score = get_form_quality_score(risk_assessment)
        quality_category = categorize_form(quality_score)

        feedback = []

        # ===== BASIC METRICS =====
        feedback.append(f"REP #{self.current_rep} - Quality: {quality_score}/100 ({quality_category})")

        if elbows:
            feedback.append(f"  Elbow: {min(elbows):.0f}° - {max(elbows):.0f}° (target: 90° at bottom)")

        if hips:
            feedback.append(f"  Hip: {min(hips):.0f}° (target: 170-185°)")

        # ===== POSITIVES =====
        positives = [f for f in risk_assessment['feedback'] if "[GOOD]" in f]
        for p in positives:
            feedback.append(f"  [GOOD] {p.replace('[GOOD] ', '')}")

        # ===== HEAD CHECK =====
        head_check = self._check_head_position(bottom_landmarks, min_elbow)
        if head_check:
            feedback.append(f"  {head_check}")
            feedback.append("  [ISSUE] Head Alignment")
            feedback.append("  [TIP] Keep neck neutral and aligned with spine")

        # ===== FLAGS =====
        for flag in risk_assessment['flags']:
            feedback.append(f"  [FLAG] {flag}")

        # ===== ISSUES =====
        for issue in risk_assessment['issues']:
            feedback.append(f"  [ISSUE] {issue.replace('_', ' ').title()}")

        # ===== CORRECTIONS =====
        corrections = [f for f in risk_assessment['feedback'] if "[GOOD]" not in f]
        for c in corrections:
            feedback.append(f"  [TIP] {c}")

        feedback.append(f"  Risk: {risk_assessment['risk_level']}")

        return feedback

    def reset(self):
        """Reset after one rep"""
        self.rep_angles = []


# ================= LEGACY SUPPORT =================
def analyze_pushup(landmarks, angles):
    """Legacy function for backwards compatibility"""
    analyzer = FormAnalyzer()
    analyzer.collect(landmarks, angles)
    return analyzer.evaluate()