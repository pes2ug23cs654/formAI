"""
Squat Temporal Engine - Smooths squat angle measurements and detects movement stages
"""

from collections import deque

class SquatTemporalEngine:
    def __init__(self, buffer_size=5):  # Slightly larger buffer for squat stability
        self.buffer = deque(maxlen=buffer_size)
        self.stage = "STANDING"
        self.last_valid = None
        self.last_raw = None

    def smooth(self, value):
        """Smooth angle measurements for stable squat detection"""
        if value is None:
            return None

        # Moderate outlier rejection (>30° jump = tracking noise)
        if self.last_raw is not None and abs(value - self.last_raw) > 30:
            if self.buffer:
                value = sum(self.buffer) / len(self.buffer)

        self.last_raw = value
        self.buffer.append(value)
        avg = sum(self.buffer) / len(self.buffer)
        self.last_valid = avg
        return avg

    def detect_stage(self, angle):
        """Detect squat movement stage based on knee angle"""
        if angle is None:
            return "UNKNOWN"

        if angle > 160:
            return "STANDING"
        elif 120 <= angle <= 160:
            return "ASCENDING"
        elif 85 <= angle <= 120:
            return "BOTTOM"
        elif 85 < angle < 140:
            return "DESCENDING"
        else:
            return "UNKNOWN"