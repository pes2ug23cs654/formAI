"""
Squat Rep Counter - Accurate Squat Detection
Counts squat reps correctly while collecting form data for analysis
"""


class SquatRepCounter:
    def __init__(self, debug=False):
        self.state = "STANDING"  # STANDING, DESCENDING, BOTTOM, ASCENDING
        self.reps = 0
        self.debug = debug
        self.frame_num = 0

        self.angles_in_rep = []   # current rep data
        self.last_rep_data = []   # 🔥 store completed rep data

        self.descent_frames = 0
        self.ascent_frames = 0
        self.bottom_frames = 0

        # 🔥 NEW: landmark tracking
        self.landmarks_buffer = []
        self.last_landmarks = []

    def update(self, knee_angle, landmarks=None):
        """Process frame or finalize on video end"""

        # ─── HANDLE VIDEO END ─────────────────────
        if knee_angle is None:
            if self.state in ["DESCENDING", "BOTTOM"] and len(self.angles_in_rep) > 15:
                self.reps += 1

                # 🔥 STORE LAST REP BEFORE RESET
                self.last_rep_data = self.angles_in_rep.copy()
                self.last_landmarks = self.landmarks_buffer.copy()

                self.angles_in_rep = []
                self.landmarks_buffer = []

                if self.debug:
                    print(f"[VIDEO END] Finalizing squat rep - total: {self.reps}", flush=True)

                return True
            return False

        self.frame_num += 1
        self.angles_in_rep.append(knee_angle)

        # 🔥 STORE LANDMARKS PER FRAME
        if landmarks:
            self.landmarks_buffer.append(landmarks)

        # ─── STATE MACHINE ────────────────────────
        if self.state == "STANDING":

            if knee_angle < 140:
                self.state = "DESCENDING"
                self.descent_frames = 0

                if self.debug and self.frame_num <= 100:
                    print(f"[F{self.frame_num}] DESCENDING ({knee_angle:.0f}°)", flush=True)

        elif self.state == "DESCENDING":
            self.descent_frames += 1

            # Bottom reached
            if 85 <= knee_angle <= 115 and self.descent_frames >= 3:
                self.state = "BOTTOM"
                self.bottom_frames = 0

                if self.debug and self.frame_num <= 500:
                    print(f"[F{self.frame_num}] BOTTOM ({knee_angle:.0f}°)", flush=True)

            # False start
            elif knee_angle > 140 and self.descent_frames < 5:
                self.state = "STANDING"

                if self.debug and self.frame_num <= 100:
                    print(f"[F{self.frame_num}] False start", flush=True)

        elif self.state == "BOTTOM":
            self.bottom_frames += 1

            if knee_angle > 120 and self.bottom_frames >= 2:
                self.state = "ASCENDING"
                self.ascent_frames = 0

                if self.debug and self.frame_num <= 500:
                    print(f"[F{self.frame_num}] ASCENDING ({knee_angle:.0f}°)", flush=True)

        elif self.state == "ASCENDING":
            self.ascent_frames += 1

            # 🔥 REP COMPLETE
            if knee_angle > 160 and self.ascent_frames >= 3:
                self.state = "STANDING"
                self.reps += 1

                # 🔥 STORE REP BEFORE RESET
                self.last_rep_data = self.angles_in_rep.copy()
                self.last_landmarks = self.landmarks_buffer.copy()

                self.angles_in_rep = []
                self.landmarks_buffer = []

                if self.debug and self.frame_num <= 500:
                    print(f"[F{self.frame_num}] SQUAT #{self.reps}", flush=True)

                return True

            # Bounce detection
            elif knee_angle < 120 and self.ascent_frames < 5:
                self.state = "BOTTOM"
                self.bottom_frames = 0

                if self.debug and self.frame_num <= 500:
                    print(f"[F{self.frame_num}] Bounce detected", flush=True)

        return False

    # 🔥 USE THIS FOR ANALYSIS
    def get_last_rep(self):
        return self.last_rep_data

    # 🔥 NEW: get last rep landmarks
    def get_last_landmarks(self):
        return self.last_landmarks

    # (keep for compatibility if needed)
    def get_rep_angles(self):
        return self.angles_in_rep

    def reset(self):
        self.angles_in_rep = []
        self.last_rep_data = []
        self.landmarks_buffer = []
        self.last_landmarks = []