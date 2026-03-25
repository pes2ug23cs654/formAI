"""
Rep Counter - Simple + Accurate Pushup Detection
Counts reps correctly while collecting form data for analysis
"""

class RepCounter:
    def __init__(self, debug=False):
        self.state = "UP"  # UP or DOWN
        self.reps = 0
        self.debug = debug
        self.frame_num = 0
        self.angles_in_rep = []  # For form analysis
        self.down_count = 0  # Frames spent descending
        
    def update(self, angle):
        """Process frame or finalize on video end"""
        # Handle video end
        if angle is None:
            if self.state == "DOWN" and len(self.angles_in_rep) > 10:
                self.reps += 1
                if self.debug:
                    print(f"[VIDEO END] Finalizing rep - total: {self.reps}", flush=True)
                self.angles_in_rep = []
                return True
            return False
        
        self.frame_num += 1
        self.angles_in_rep.append(angle)
        
        # OPTIMIZED FOR ALL REP SPEEDS (even 30+ reps/min)
        if self.state == "UP":
            # Descent trigger: 110° (catches even fast shallow reps)
            if angle < 110:
                self.state = "DOWN"
                self.down_count = 0
                if self.debug and self.frame_num <= 100:
                    print(f"[F{self.frame_num}] DOWN ({angle:.0f}°)", flush=True)
                    
        elif self.state == "DOWN":
            # Count frames in descent phase
            self.down_count += 1
            
            # Quick ascent: needs min 2 frames down + angle >155°
            # This catches rapid reps
            if self.down_count >= 2 and angle > 155:
                self.state = "UP"
                self.reps += 1
                if self.debug and self.frame_num <= 500:
                    print(f"[F{self.frame_num}] UP - REP #{self.reps} (speed: {self.down_count} frames down)", flush=True)
                self.angles_in_rep = []
                return True  # Signal rep completed
        
        return False
    
    def get_rep_angles(self):
        """Return angles collected during rep"""
        return self.angles_in_rep
    
    def reset(self):
        """Reset for next rep"""
        self.angles_in_rep = []

