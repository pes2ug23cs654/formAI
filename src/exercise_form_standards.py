"""
COMPREHENSIVE FORM STANDARDS FOR ALL 7 EXERCISES
Defines correct ranges, common mistakes, and human variations
"""

import numpy as np


class FormStandards:
    """Base class for exercise-specific form requirements"""
    
    @staticmethod
    def is_within_range(value, min_val, max_val, tolerance=0):
        """Check if value is within acceptable range with tolerance"""
        if value is None:
            return None
        return (min_val - tolerance) <= value <= (max_val + tolerance)


class PushupFormStandards(FormStandards):
    """Push-up form validation with human variation tolerance"""
    
    # ─── ELBOW ANGLE (Primary) ───────────────────────
    ELBOW_DEPTH_OPTIMAL = 90      # Perfect at bottom
    ELBOW_DEPTH_MIN = 60           # Minimum (hyperextension risk)
    ELBOW_DEPTH_MAX = 120          # Maximum (not deep enough)
    ELBOW_DEPTH_LENIENT_MIN = 55   # Very lenient for mobility issues
    ELBOW_DEPTH_LENIENT_MAX = 130  # Lenient for limited mobility
    
    ELBOW_TOP_MIN = 150            # Minimum extension at top
    ELBOW_TOP_LENIENT_MIN = 140    # Lenient for arm length differences
    
    # ─── HIP ANGLE (Body Alignment) ───────────────
    HIP_OPTIMAL = 175              # Perfect neutral spine
    HIP_MIN = 165                  # Minimum (hips sagging)
    HIP_MAX = 190                  # Maximum (pike position)
    HIP_LENIENT_MIN = 155          # Lenient (some sagging)
    HIP_LENIENT_MAX = 200          # Lenient (some pike)
    
    # ─── KNEE ANGLE (Leg Extension) ──────────────
    KNEE_OPTIMAL = 180             # Fully locked
    KNEE_MIN = 170                 # Nearly straight
    KNEE_LENIENT_MIN = 160         # Lenient (slight bend OK)
    
    # ─── SHOULDER ANGLE (Arm Position) ───────────
    SHOULDER_WIDTH_OPTIMAL = 90    # 90 degrees from body
    SHOULDER_WIDTH_MIN = 60        # Too narrow
    SHOULDER_WIDTH_MAX = 120       # Too wide


class SquatFormStandards(FormStandards):
    """Squat form validation with depth types"""
    
    # ─── KNEE ANGLE (Primary) ─────────────────────
    KNEE_DEPTH_FULL = 75           # Full squat (ass-to-grass)
    KNEE_DEPTH_PARALLEL = 90       # Parallel (hips level with knees)
    KNEE_DEPTH_ATG = 60            # Below parallel (advanced)
    
    KNEE_MIN = 50                  # Minimum reasonable depth
    KNEE_MAX = 140                 # Maximum (not moving)
    KNEE_LENIENT_MIN = 45          # Lenient for ankle mobility
    KNEE_LENIENT_MAX = 150         # Lenient upper bound
    
    # ─── HIP ANGLE (Back Position) ─────────────────
    HIP_MIN = 60                   # Hips too high (not deep enough)
    HIP_MAX = 180                  # Hips fully extended (top)
    
    # ─── HIP VALGUS DETECTION (Knee Cave) ────────
    HIP_VALGUS_THRESHOLD = 15      # Max inward knee collapse (degrees)
    
    # ─── ANKLE DORSIFLEXION ──────────────────────
    # Ankles should show with sufficient dorsiflexion (forward lean OK)
    ANKLE_POSITION_FORWARD_LEAN = 25  # Degrees forward lean acceptable
    
    # ─── BACK ANGLE ──────────────────────────────
    BACK_NEUTRAL_ANGLE = 50        # Relative to ground
    BACK_ROUNDING_TOLERANCE = 15   # Degrees of rounding acceptable


class LungeFormStandards(FormStandards):
    """Lunge form validation"""
    
    # ─── FRONT KNEE (Primary) ────────────────────
    FRONT_KNEE_OPTIMAL = 90        # 90 degrees at bottom
    FRONT_KNEE_MIN = 70            # Too shallow
    FRONT_KNEE_MAX = 110           # Too deep
    FRONT_KNEE_LENIENT_MIN = 60    # Lenient shallow
    FRONT_KNEE_LENIENT_MAX = 120   # Lenient deep
    
    # ─── BACK KNEE ──────────────────────────────
    BACK_KNEE_OPTIMAL = 30         # Near ground but not touching
    BACK_KNEE_MIN = 10             # Too high (shallow step)
    BACK_KNEE_MAX = 45             # Too low (touching/discomfort)
    
    # ─── TORSO ALIGNMENT ───────────────────────
    TORSO_UPRIGHT_OPTIMAL = 85     # Upright position (relative to ground)
    TORSO_FORWARD_LEAN_MAX = 20    # Forward lean tolerance
    TORSO_BACK_LEAN_MAX = 10       # Back lean (should be minimal)
    
    # ─── KNEE TRACKING ─────────────────────────
    FRONT_KNEE_OVER_ANKLE = 5      # Centimeters - knee over ankle
    FRONT_KNEE_CAVE_THRESHOLD = 12 # Degrees - knee inward collapse
    
    # ─── STRIDE LENGTH ─────────────────────────
    STRIDE_CONSISTENCY = 0.2        # Allow 20% variation between lunges


class BicepCurlFormStandards(FormStandards):
    """Bicep curl form validation"""
    
    # ─── ELBOW ANGLE (Primary) ───────────────────
    ELBOW_BOTTOM_OPTIMAL = 170     # Full extension at bottom
    ELBOW_BOTTOM_MIN = 155         # Minimum extension
    ELBOW_BOTTOM_LENIENT_MIN = 145 # Lenient (slight bend OK)
    
    ELBOW_TOP_OPTIMAL = 35         # Full contraction at top
    ELBOW_TOP_MAX = 50             # Maximum angle at top
    ELBOW_TOP_LENIENT_MAX = 60     # Lenient (less range)
    
    # ─── UPPER ARM STABILITY ─────────────────────
    SHOULDER_HORIZONTAL_PIN = 5    # Degrees - elbow shouldn't move horizontally
    UPPER_ARM_FORWARD_MAX = 15     # Forward swing tolerance
    UPPER_ARM_BACKWARD_MAX = 5     # Backward swing (minimal)
    
    # ─── MOMENTUM DETECTION ──────────────────────
    SPEED_THRESHOLD_FAST = 3       # Degrees per frame (jerky motion)
    SPEED_THRESHOLD_SLOW = 0.3     # Degrees per frame (too slow)
    
    # ─── WRIST POSITION ──────────────────────────
    WRIST_NEUTRAL_OPTIMAL = 0      # Neutral
    WRIST_SUPINATION_TOLERANCE = 20  # Degrees of supination
    WRIST_PRONATION_TOLERANCE = 10   # Degrees of pronation (less OK)


class ShoulderPressFormStandards(FormStandards):
    """Shoulder press form validation"""
    
    # ─── ELBOW ANGLE (Primary) ───────────────────
    ELBOW_BOTTOM_OPTIMAL = 90      # 90° at shoulder height start
    ELBOW_BOTTOM_MIN = 70          # Minimum (elbows dropped)
    ELBOW_BOTTOM_LENIENT_MIN = 60  # Lenient
    
    ELBOW_TOP_OPTIMAL = 170        # Full lockout extension
    ELBOW_TOP_MIN = 160            # Minimum lockout
    ELBOW_TOP_LENIENT_MIN = 150    # Lenient (slight bend)
    
    # ─── LOCKOUT HEIGHT ──────────────────────────
    LOCKOUT_FULL = 180             # Fully overhead
    LOCKOUT_FORWARD_LEAN_MAX = 15  # Forward lean at top
    LOCKOUT_BACKWARD_LEAN_MAX = 10 # Back lean (should be neutral)
    
    # ─── LOWER BACK COMPENSATION ─────────────────
    HIP_ANGLE_OPTIMAL = 175        # Neutral spine
    HIP_ANGLE_MIN = 160            # Minimum (arching back)
    HIP_ANGLE_LENIENT_MIN = 150    # Lenient
    
    # ─── LOWER BACK SWAY ─────────────────────────
    LOWER_BACK_SWAY_MAX = 10       # Degrees of sway
    
    # ─── SHOULDER PACKING ───────────────────────
    # Shoulders should pack down, not shrug up
    SHOULDER_ELEVATION_MAX = 8     # Degrees shrug up (minimal)


class SitupFormStandards(FormStandards):
    """Sit-up form validation"""
    
    # ─── HIP ANGLE (Primary) ──────────────────────
    HIP_FLAT_OPTIMAL = 30          # Lying flat
    HIP_FLATISH_MIN = 25           # Nearly flat
    HIP_BOTTOM_THRESHOLD = 40      # Below this = flat
    
    HIP_CONTRACTED_OPTIMAL = 70    # Fully contracted
    HIP_CONTRACTED_MIN = 55        # Minimum contraction
    HIP_CONTRACTED_LENIENT_MAX = 85  # Lenient (less range)
    
    # ─── TOTAL RANGE ──────────────────────────────
    HIP_RANGE_OPTIMAL = 50         # 30° to 80°
    HIP_RANGE_MIN = 35             # Minimum range
    
    # ─── NECK POSITION ────────────────────────────
    NECK_NEUTRAL = 0               # Aligned with spine
    NECK_FLEXION_MIN = -30         # Degree neck can flex
    NECK_FLEXION_MAX = 15          # Maximum safe flexion
    
    HEAD_CHIN_DISTANCE = 5         # Inches - maintain distance from chest
    
    # ─── LOWER BACK CONTACT ──────────────────────
    # Lower back should stay on ground until final portion
    LOWER_BACK_ENGAGED_UNTIL = 45  # Hip angle - then lower back lives
    
    # ─── MOMENTUM ────────────────────────────────
    SITUP_SPEED_FAST = 2.5         # Degrees per frame (too jerky)
    SITUP_SPEED_SLOW = 0.2         # Degrees per frame (too slow)


class MountainClimberFormStandards(FormStandards):
    """Mountain climber form validation"""
    
    # ─── PLANK POSITION ────────────────────────────
    HIP_OPTIMAL = 175              # Neutral spine in plank
    HIP_SAGGING_THRESHOLD = 150    # Below this = sagging
    HIP_PIKE_THRESHOLD = 195       # Above this = pike
    
    # ─── KNEE DRIVE (Primary) ──────────────────────
    KNEE_DRIVE_HEIGHT_OPTIMAL = 80 # Angle - knee high (chest level)
    KNEE_DRIVE_MIN = 60            # Minimum drive height
    KNEE_DRIVE_MAX = 95            # Maximum (full contraction)
    
    # ─── KNEE EXTENSION ────────────────────────────
    KNEE_EXTENDED_OPTIMAL = 170    # Back leg fully extended
    KNEE_EXTENDED_MIN = 155        # Minimum extension
    
    # ─── HIP STABILITY ─────────────────────────────
    HIP_SWAY_THRESHOLD = 8         # Degrees - max side to side
    HIP_ROTATION_THRESHOLD = 10    # Degrees - max rotation
    
    # ─── HAND POSITION ────────────────────────────
    HAND_WIDTH_OPTIMAL = 90        # 90 degrees apart
    HAND_WIDTH_MIN = 70            # Too narrow
    HAND_WIDTH_MAX = 110           # Too wide
    
    # ─── RHYTHM/CADENCE ───────────────────────────
    CLIMB_SPEED_MIN = 0.5          # Reps per second minimum
    CLIMB_SPEED_MAX = 3.0          # Reps per second maximum
    CLIMB_CONSISTENCY_TOLERANCE = 0.3  # Variation in pace
