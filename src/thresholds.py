"""
Calibrated Thresholds for Push-up Detection
Based on Phase 1 calibration analysis of input.mp4
Values derived from actual human movement physics, not textbook parameters
"""

# ANGLE THRESHOLDS (Elbow Joint)
# Calibration Source: input.mp4 analysis + long.mp4 observation
# Actual observed range: input.mp4 min=51.6° max=175.9°, long.mp4 min=75.5° max=168.7°
# Setting thresholds to handle both videos

TOP_ANGLE = 155.0              # Target at top - set to 155° to match long.mp4 max of 159.5°  
BOTTOM_ANGLE = 85.0            # Target at bottom - matches long.mp4 min of 75-83°
DESCENT_TRIGGER = 140.0        # Angle to start descent tracking
ASCENT_THRESHOLD = 114.0       # Midpoint to trigger ascent state

# DEPTH EVALUATION
MIN_DEPTH_GOOD = 65.0          # Minimum angle for "good depth"
MAX_DEPTH_SHALLOW = 95.0       # Maximum angle to avoid being "not deep enough"

# BODY ALIGNMENT (Shoulder-Hip-Ankle angles)
BODY_ANGLE_MIN = 160.0         # Minimum acceptable plank angle
BODY_ANGLE_MAX = 195.0         # Maximum acceptable plank angle

# MOTION VALIDATION (for real vs false pushups)
MIN_VALID_RANGE = 60.0         # Minimum angle range for a valid rep (80° - 20°)
MAX_HIP_DEVIATION = 80.0       # Max pixel deviation from shoulder-ankle line (was 40px)
MIN_MOTION_VARIANCE = 15.0     # Minimum variance to distinguish from static position (kneeling)

# TEMPORAL SMOOTHING
SMOOTHING_BUFFER_SIZE = 5      # Moving average window (frames)
MOTION_HISTORY_SIZE = 15       # Angle history for trend detection (frames)

# ACTIVATION GATES
STRICT_PLANK_MODE_DURATION = 120  # Frames to require plank activation (at 60fps: 2 seconds)
PLANK_ACTIVATION_THRESHOLD = 8    # Consecutive frames needed for plank activation

# STATE MACHINE SAFETY
MAX_STUCK_FRAMES = 100         # Frames before auto-reset if stuck in DESCENT/BOTTOM
