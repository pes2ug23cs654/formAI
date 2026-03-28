from collections import deque

class TemporalEngine:
    def __init__(self, buffer_size=3):  # Reduced to 3 for fast rep detection (30+ reps/min)
        self.buffer = deque(maxlen=buffer_size)
        self.stage = "TOP"
        self.last_valid = None
        self.last_raw = None

    def smooth(self, value):
        """Minimal smoothing - optimized for fast reps"""
        if value is None:
            return None
        
        # Keep true fast-motion transitions; reject only extreme jumps.
        if self.last_raw is not None and abs(value - self.last_raw) > 70:
            if self.buffer:
                value = sum(self.buffer) / len(self.buffer)
        
        self.last_raw = value
        self.buffer.append(value)
        avg = sum(self.buffer) / len(self.buffer)
        self.last_valid = avg
        return avg

    def detect_stage(self, angle):
        if angle is None:
            return "UNKNOWN"

        if angle > 150:
            return "TOP"
        elif angle < 90:
            return "BOTTOM"
        elif angle < 140:
            return "DESCENT"
        else:
            return "ASCENT"