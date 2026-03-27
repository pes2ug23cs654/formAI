import numpy as np


class SquatAnalyzer:
    def __init__(self):
        pass

    # ================= ANGLE FUNCTION =================
    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        angle = np.degrees(
            np.arccos(
                np.dot(ba, bc) /
                (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            )
        )
        return angle

    # ================= MAIN ANALYSIS =================
    def analyze(self, rep_angles, landmarks_seq=None):
        issues = []
        feedback = []

        # ---------------- DEPTH ----------------
        bottom = [a for a in rep_angles if a < 120]

        if len(bottom) > 5:
            knee_angle = float(np.median(bottom))
        else:
            knee_angle = float(min(rep_angles)) if rep_angles else 180.0

        if knee_angle < 75:
            feedback.append("Excellent depth")
        elif knee_angle < 95:
            feedback.append("Good depth")
        else:
            issues.append("shallow_squat")

        # ---------------- ADVANCED ANALYSIS ----------------
        if landmarks_seq:

            valgus_count = 0
            forward_lean_count = 0
            poor_hinge_count = 0

            for lm in landmarks_seq:
                try:
                    shoulder = lm['left_shoulder']
                    hip = lm['left_hip']
                    knee = lm['left_knee']
                    ankle = lm['left_ankle']

                    # -------- KNEE COLLAPSE --------
                    if knee['x'] < ankle['x'] + 0.02:
                        valgus_count += 1

                    # -------- TORSO ANGLE --------
                    torso_angle = self.calculate_angle(
                        [shoulder['x'], shoulder['y']],
                        [hip['x'], hip['y']],
                        [knee['x'], knee['y']]
                    )

                    # forward lean detection
                    if torso_angle < 150:
                        forward_lean_count += 1

                    # -------- HIP HINGE --------
                    # compare hip vs knee horizontal displacement
                    if abs(hip['x'] - knee['x']) < 0.04:
                        poor_hinge_count += 1

                except Exception:
                    continue

            n = len(landmarks_seq)

            # -------- APPLY THRESHOLDS --------
            if valgus_count > n * 0.25:
                issues.append("knee_collapse")

            if forward_lean_count > n * 0.3:
                issues.append("forward_lean")

            if poor_hinge_count > n * 0.3:
                issues.append("poor_hip_hinge")

        # ---------------- SCORING ----------------
        score = 100

        if "shallow_squat" in issues:
            score -= 30
        if "knee_collapse" in issues:
            score -= 20
        if "forward_lean" in issues:
            score -= 15
        if "poor_hip_hinge" in issues:
            score -= 15

        score = max(score, 50)

        # ---------------- FEEDBACK ----------------
        if "forward_lean" in issues:
            feedback.append("Keep chest up")

        if "poor_hip_hinge" in issues:
            feedback.append("Push hips back more")

        if "knee_collapse" in issues:
            feedback.append("Keep knees aligned with toes")

        return {
            "score": score,
            "issues": issues,
            "feedback": feedback,
            "knee_angle": knee_angle
        }