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

    # ================= BACK ANGLE =================
    def get_back_angle(self, shoulder, hip):
        dx = shoulder['x'] - hip['x']
        dy = shoulder['y'] - hip['y']
        angle = np.degrees(np.arctan2(dy, dx))
        return abs(angle)

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

            forward_lean_count = 0
            poor_hinge_count = 0
            knee_forward_count = 0
            bottom_frames = 0

            for lm, angle in zip(landmarks_seq, rep_angles):
                try:
                    shoulder = lm[11]
                    hip = lm[23]
                    knee = lm[25]
                    ankle = lm[27]

                    # -------- BACK POSTURE --------
                    back_angle = self.get_back_angle(shoulder, hip)

                    if back_angle < 50:
                        forward_lean_count += 1

                    # -------- ONLY CHECK NEAR BOTTOM --------
                    if angle < 100:
                        bottom_frames += 1

                        # -------- HIP HINGE (FIXED 🔥) --------
                        body_length = np.linalg.norm([
                            hip['x'] - shoulder['x'],
                            hip['y'] - shoulder['y']
                        ])

                        hip_shift = abs(hip['x'] - ankle['x'])
                        ratio = hip_shift / (body_length + 1e-6)

                        print(f"[HINGE DEBUG - BOTTOM] ratio: {ratio:.3f}")

                        if ratio < 0.15:
                            poor_hinge_count += 1

                    # -------- KNEE TRAVEL --------
                    if knee['x'] > ankle['x'] + 40:
                        knee_forward_count += 1

                except Exception:
                    continue

            n = len(landmarks_seq)

            # -------- APPLY THRESHOLDS --------
            if forward_lean_count > n * 0.3:
                issues.append("forward_lean")

            if bottom_frames > 0 and poor_hinge_count > bottom_frames * 0.4:
                issues.append("poor_hip_hinge")

            if knee_forward_count > n * 0.3:
                issues.append("knees_too_forward")

        # ---------------- SCORING ----------------
        score = 100

        if "shallow_squat" in issues:
            score -= 25
        if "poor_hip_hinge" in issues:
            score -= 20
        if "forward_lean" in issues:
            score -= 15
        if "knees_too_forward" in issues:
            score -= 10

        score = max(score, 50)

        # ---------------- FEEDBACK ----------------
        if "shallow_squat" in issues:
            feedback.append("Go deeper (target <90° knee angle)")
        else:
            feedback.append("Good depth")

        if "poor_hip_hinge" in issues:
            feedback.append("Push hips back more")
        else:
            feedback.append("Good hip hinge")

        if "forward_lean" in issues:
            feedback.append("Keep chest up")
        else:
            feedback.append("Good posture")

        if "knees_too_forward" in issues:
            feedback.append("Don't push knees too far forward")

        if not issues:
            feedback.append("Excellent squat form")

        return {
            "score": score,
            "issues": issues,
            "feedback": feedback,
            "knee_angle": knee_angle
        }