import numpy as np


class RepAnalyzer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.elbows = []
        self.hips = []
        self.knees = []
        self.head_flags = []
        self.hand_flags = []

    def collect(self, landmarks, angles):
        if angles is None or landmarks is None:
            return

        elbow = angles.get('left_elbow') or angles.get('right_elbow')
        hip   = angles.get('left_hip') or angles.get('right_hip')
        knee  = angles.get('left_knee') or angles.get('right_knee')

        # ── SMOOTH ELBOW (IMPORTANT 🔥)
        if elbow:
            if len(self.elbows) > 0:
                elbow = 0.7 * self.elbows[-1] + 0.3 * elbow
            self.elbows.append(elbow)

        if hip:
            self.hips.append(hip)

        if knee:
            self.knees.append(knee)

        # ── HEAD CHECK
        if self.is_head_dropping(landmarks):
            self.head_flags.append(True)

        # ── HAND CHECK
        if self.is_hand_far(landmarks):
            self.hand_flags.append(True)

    # ── HEAD DETECTION (RELATIVE)
    def is_head_dropping(self, landmarks):
        try:
            nose = landmarks[0]
            shoulder = landmarks[11]
            hip = landmarks[23]

            torso_height = abs(hip['y'] - shoulder['y'])
            drop = nose['y'] - shoulder['y']

            return drop > torso_height * 0.3

        except:
            return False

    # ── HAND DETECTION (ELBOW-BASED)
    def is_hand_far(self, landmarks):
        try:
            wrist = landmarks[15]
            shoulder = landmarks[11]
            elbow = landmarks[13]

            shoulder_to_elbow = abs(elbow['x'] - shoulder['x'])
            elbow_to_wrist = abs(wrist['x'] - elbow['x'])

            return elbow_to_wrist > shoulder_to_elbow * 1.7

        except:
            return False

    def evaluate(self):
        feedback = []

        if not self.elbows:
            return ["No valid data"]

        # 🔥 ROBUST DEPTH (PERCENTILE FIX)
        min_elbow = np.percentile(self.elbows, 10)
        max_elbow = np.percentile(self.elbows, 90)

        avg_hip = sum(self.hips)/len(self.hips) if self.hips else None

        # ── DEPTH ─────────────────────
        if min_elbow > 110:
            feedback.append("Too shallow")
        elif min_elbow < 45:   # adjusted threshold
            feedback.append("Too deep (joint stress)")
        else:
            feedback.append("Good depth")

        # ── EXTENSION ────────────────
        if max_elbow < 150:
            feedback.append("Did not fully extend arms")

        # ── BODY ALIGNMENT ───────────
        if avg_hip:
            if avg_hip < 150:
                feedback.append("Hips sagging")
            elif avg_hip > 190:
                feedback.append("Hips too high")
            else:
                feedback.append("Good body alignment")

        # ── LEG EXTENSION (SEVERE ONLY IF REALLY BENT)
        if self.knees:
            min_knee = np.percentile(self.knees, 20)
            if min_knee < 155:
                feedback.append("Knees bent too much")

        # ── HEAD (ONLY IF CONSISTENT)
        if len(self.head_flags) > len(self.elbows) * 0.7:
            feedback.append("Head dropping")

        # ── HANDS (ONLY IF CONSISTENT)
        if len(self.hand_flags) > len(self.elbows) * 0.7:
            feedback.append("Hands too far from shoulders")

        return feedback