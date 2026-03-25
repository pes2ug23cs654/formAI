"""
Motion Validator - Detects real push-up motion vs false positives
Prevents counting: kneeling, arm waves, idle movement
"""

from collections import deque
from .thresholds import BOTTOM_ANGLE, TOP_ANGLE

class MotionValidator:
    def __init__(self, buffer_size=15):
        """
        Tracks angle history to validate real push-up motion
        
        Args:
            buffer_size: Number of frames to track for motion analysis
        """
        self.angle_buffer = deque(maxlen=buffer_size)
        self.frame_count = 0
        self.descent_start_angle = None
        self.descent_frames = 0
        self.ascent_frames = 0
        
    def add_angle(self, angle):
        """Add angle sample to history"""
        if angle is not None:
            self.angle_buffer.append(angle)
            self.frame_count += 1
    
    def is_descending(self, angle, threshold=5):
        """Check if angle is consistently decreasing (descent motion)"""
        if len(self.angle_buffer) < 3:
            return False
        
        recent = list(self.angle_buffer)[-3:]
        # True descent: recent angles trending downward
        return recent[-1] < recent[0] - threshold
    
    def is_ascending(self, angle, threshold=5):
        """Check if angle is consistently increasing (ascent motion)"""
        if len(self.angle_buffer) < 3:
            return False
        
        recent = list(self.angle_buffer)[-3:]
        # True ascent: recent angles trending upward
        return recent[-1] > recent[0] + threshold
    
    def get_angle_range(self):
        """Get min and max angles in buffer"""
        if not self.angle_buffer:
            return None, None
        return min(self.angle_buffer), max(self.angle_buffer)
    
    def is_valid_pushup_cycle(self):
        """
        Determine if angle history represents real push-up motion
        
        Real push-up:
        - MIN angle < 70° (goes to bottom)
        - MAX angle > 150° (goes to top)
        - Range > 80° (significant motion)
        - Not stuck at one angle (variance > 10°)
        """
        if len(self.angle_buffer) < 5:
            return False
        
        min_angle, max_angle = self.get_angle_range()
        if min_angle is None:
            return False
        
        angle_range = max_angle - min_angle
        
        # Real pushup has large range
        if angle_range < 80:
            return False
        
        # Bottom must be deep enough
        if min_angle > BOTTOM_ANGLE + 10:  # Allow 10° tolerance
            return False
        
        # Top must be extended enough
        if max_angle < TOP_ANGLE - 10:  # Allow 10° tolerance
            return False
        
        return True
    
    def is_static_position(self, threshold=15):
        """
        Detect if position is static (kneeling, idle)
        Static = very little angle variation
        """
        if len(self.angle_buffer) < 5:
            return False
        
        min_angle, max_angle = self.get_angle_range()
        return (max_angle - min_angle) < threshold
    
    def is_valid_descent_motion(self, current_angle):
        """
        Validate that we're in descent (angle decreasing toward bottom)
        Just check trend - current should be less than recent average
        """
        if len(self.angle_buffer) < 3:
            return True  # Not enough history yet
        
        recent = list(self.angle_buffer)[-3:]
        average = sum(recent) / len(recent)
        
        # Descending if we're trending toward smaller values
        return current_angle <= average + 5
    
    def is_valid_ascent_motion(self, current_angle):
        """
        Validate that we're in ascent (angle increasing toward top)
        Just check trend - current should be greater than recent average
        """
        if len(self.angle_buffer) < 3:
            return True  # Not enough history yet
        
        recent = list(self.angle_buffer)[-3:]
        average = sum(recent) / len(recent)
        
        # Ascending if we're trending toward larger values
        return current_angle >= average - 5
    
    def reset(self):
        """Clear history"""
        self.angle_buffer.clear()
        self.frame_count = 0
        self.descent_start_angle = None
        self.descent_frames = 0
        self.ascent_frames = 0
