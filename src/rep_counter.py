"""
Rep Counter - Simple + Accurate Pushup Detection
Counts reps correctly while collecting form data for analysis
"""

class RepCounter:
    def __init__(
        self,
        debug=False,
        descent_trigger=110.0,
        ascent_threshold=155.0,
        min_down_frames=2,
        fast_ascent_margin=8.0,
        require_initial_lockout=False,
        initial_lockout_frames=2,
    ):
        self.state = "UP"  # UP or DOWN
        self.reps = 0
        self.debug = debug
        self.frame_num = 0
        self.angles_in_rep = []  # For form analysis
        self.down_count = 0  # Frames spent descending
        self.prev_angle = None
        self.min_angle_in_down = None
        self.last_completed_rep_angles = []
        self.descent_trigger = descent_trigger
        self.ascent_threshold = ascent_threshold
        self.min_down_frames = min_down_frames
        self.fast_ascent_margin = float(fast_ascent_margin)
        self.require_initial_lockout = bool(require_initial_lockout)
        self.initial_lockout_frames = max(1, int(initial_lockout_frames))
        self.initial_lockout_seen = not self.require_initial_lockout
        self.initial_lockout_streak = 0

    def set_thresholds(self, descent_trigger=None, ascent_threshold=None, min_down_frames=None):
        """Update state-machine thresholds at runtime (used by calibration)."""
        if descent_trigger is not None:
            self.descent_trigger = float(descent_trigger)
        if ascent_threshold is not None:
            self.ascent_threshold = float(ascent_threshold)
        if min_down_frames is not None:
            self.min_down_frames = int(min_down_frames)
        
    def update(self, angle):
        """Process frame or finalize on video end"""
        # Handle video end
        if angle is None:
            if self.state == "DOWN" and len(self.angles_in_rep) > 10:
                self.reps += 1
                self.last_completed_rep_angles = self.angles_in_rep[:]
                if self.debug:
                    print(f"[VIDEO END] Finalizing rep - total: {self.reps}", flush=True)
                self.angles_in_rep = []
                return True
            return False
        
        self.frame_num += 1
        self.angles_in_rep.append(angle)

        # Prevent false first rep when clip starts mid-rep: require top lockout first.
        if not self.initial_lockout_seen:
            if angle >= self.ascent_threshold:
                self.initial_lockout_streak += 1
            else:
                self.initial_lockout_streak = 0

            if self.initial_lockout_streak >= self.initial_lockout_frames:
                self.initial_lockout_seen = True
                self.state = "UP"
                self.down_count = 0
                self.angles_in_rep = []
                if self.debug:
                    print(f"[F{self.frame_num}] INITIAL LOCKOUT READY", flush=True)

            self.prev_angle = angle
            return False
        
        # OPTIMIZED FOR ALL REP SPEEDS (even 30+ reps/min)
        if self.state == "UP":
            # Descent trigger catches transition from top toward bottom.
            if angle < self.descent_trigger:
                self.state = "DOWN"
                self.down_count = 0
                self.min_angle_in_down = angle
                if self.debug and self.frame_num <= 100:
                    print(f"[F{self.frame_num}] DOWN ({angle:.0f}°)", flush=True)
                    
        elif self.state == "DOWN":
            # Count frames in descent phase
            self.down_count += 1
            self.min_angle_in_down = angle if self.min_angle_in_down is None else min(self.min_angle_in_down, angle)

            # Fast rebound detection: when fps is low or motion is very quick,
            # lockout may be missed by a few degrees on sampled frames.
            angle_rise = 0.0 if self.prev_angle is None else (angle - self.prev_angle)
            near_lockout = angle >= (self.ascent_threshold - self.fast_ascent_margin)
            clearly_bottomed = self.min_angle_in_down is not None and self.min_angle_in_down < self.descent_trigger
            fast_rebound = near_lockout and angle_rise >= 4.0 and clearly_bottomed
            
            # Quick ascent threshold catches rapid reps while avoiding flicker.
            if self.down_count >= self.min_down_frames and (angle > self.ascent_threshold or fast_rebound):
                self.state = "UP"
                self.reps += 1
                self.last_completed_rep_angles = self.angles_in_rep[:]
                if self.debug and self.frame_num <= 500:
                    print(f"[F{self.frame_num}] UP - REP #{self.reps} (speed: {self.down_count} frames down)", flush=True)
                self.angles_in_rep = []
                self.min_angle_in_down = None
                self.prev_angle = angle
                return True  # Signal rep completed

        self.prev_angle = angle
        
        return False
    
    def get_rep_angles(self):
        """Return angles collected during rep"""
        return self.angles_in_rep

    def get_last_completed_rep_angles(self):
        """Return primary-angle sequence for most recently completed rep."""
        return self.last_completed_rep_angles
    
    def reset(self):
        """Reset for next rep"""
        self.angles_in_rep = []
        self.prev_angle = None
        self.min_angle_in_down = None
        self.last_completed_rep_angles = []
        self.initial_lockout_streak = 0
        self.initial_lockout_seen = not self.require_initial_lockout

