"""
EXERCISE-SPECIFIC REP VALIDATION LOGIC
Determines whether to count a rep based on comprehensive form rules
"""


class ExerciseRepValidator:
    """Validates if a rep should be counted based on exercise-specific rules"""
    
    @staticmethod
    def should_count_pushup(rep_score, feedback):
        """Push-up rep validation - strict due to common false positives"""
        issues = set(rep_score.get("issues", []))
        invalid_reasons = set(rep_score.get("invalid_reasons", []))
        warnings = set(rep_score.get("warnings", []))
        score = rep_score.get("score", 10)
        no_data_keys = {"no_valid_data", "not_pushup_pattern", "no_body_alignment_data"}
        
        # REJECT CONDITIONS
        # Bent knees = not a real pushup
        if "bent_knees" in issues or "bent_knees" in invalid_reasons:
            return False, "Knees must be locked for valid rep"

        # Keep strict rejection only for severe hip sag (issue-level), not light warning noise.
        if "hip_sag" in issues:
            return False, "Posture invalid: keep back straight (hips sagging)"

        # Pike warning alone should not invalidate an otherwise good rep.
        if "pike_position" in issues:
            return False, "Posture invalid: keep body in a straight plank"
        
        # Sagging hips + poor score = form breakdown
        if "sagging_hips" in issues and score < 6:
            return False, "Hips sagging too much - maintain straight body"
        
        # No body alignment = not a pushup
        if issues.intersection(no_data_keys) or invalid_reasons.intersection(no_data_keys):
            return False, "Cannot verify body alignment"
        
        # Hyperextension = injury risk
        if "hyperextension" in issues:
            return False, "Elbows hyperextending - adjust depth"
        
        # Too many major issues
        if len(issues) >= 4 or score < 4.5:
            return False, "Form breakdown - too many issues"
        
        # ACCEPT CONDITIONS
        if score >= 5.5:  # Lenient threshold
            return True, "Rep counted - acceptable form"
        
        return rep_score.get("is_valid", True), "Form analysis"
    
    @staticmethod
    def should_count_squat(rep_score, feedback):
        """Squat rep validation"""
        issues = set(rep_score.get("invalid_reasons", []))
        warnings = set(rep_score.get("warnings", []))
        score = rep_score.get("score", 10)
        no_data_keys = {"no_valid_data", "not_pushup_pattern", "no_body_alignment_data"}
        
        # REJECT CONDITIONS
        # No depth data = can't verify
        if issues.intersection(no_data_keys) or score < 4:
            return False, "Cannot verify depth"
        
        # Extremely shallow
        if "very_shallow_depth" in issues:
            return False, "Squat too shallow - needs more depth"
        
        # Multiple major issues
        if len(issues) >= 3:
            return False, "Too many form issues"
        
        # ACCEPT CONDITIONS
        # Allow partial reps but penalize in score
        if score >= 5:
            return True, "Rep counted"
        
        return False, "Form too poor"
    
    @staticmethod
    def should_count_lunge(rep_score, feedback):
        """Lunge rep validation"""
        issues = set(rep_score.get("invalid_reasons", []))
        score = rep_score.get("score", 10)
        no_data_keys = {"no_valid_data", "not_pushup_pattern", "no_body_alignment_data"}
        
        # REJECT CONDITIONS
        if issues.intersection(no_data_keys) or score < 4:
            return False, "Cannot verify form"
        
        # Unstable lunge
        if "unstable_lunge" in issues and score < 60:
            return False, "Lunge too unstable"
        
        if len(issues) >= 3:
            return False, "Too many form issues"
        
        # ACCEPT CONDITIONS
        if score >= 5:
            return True, "Rep counted"
        
        return False, "Form too poor"
    
    @staticmethod
    def should_count_bicep_curl(rep_score, feedback):
        """Bicep curl rep validation"""
        issues = set(rep_score.get("invalid_reasons", []))
        score = rep_score.get("score", 10)
        no_data_keys = {"no_valid_data", "not_pushup_pattern", "no_body_alignment_data"}
        
        # REJECT CONDITIONS
        if issues.intersection(no_data_keys) or score < 4:
            return False, "Cannot verify form"
        
        # No full extension AT LEAST
        if "no_full_extension" in issues:
            return False, "Must fully extend arms at bottom"
        
        # If using too much momentum
        if "momentum_cheat" in issues and "limited_rom" in issues:
            return False, "Using momentum instead of muscle"
        
        if len(issues) >= 3:
            return False, "Too many form issues"
        
        # ACCEPT CONDITIONS
        if score >= 5:
            return True, "Rep counted"
        
        return False, "Form too poor"
    
    @staticmethod
    def should_count_shoulder_press(rep_score, feedback):
        """Shoulder press rep validation"""
        issues = set(rep_score.get("invalid_reasons", []))
        score = rep_score.get("score", 10)
        no_data_keys = {"no_valid_data", "not_pushup_pattern", "no_body_alignment_data"}
        
        # REJECT CONDITIONS
        if issues.intersection(no_data_keys) or score < 4:
            return False, "Cannot verify form"
        
        # No lockout = not a valid press
        if "no_lockout" in issues:
            return False, "Must lock out fully overhead"
        
        # Extreme lower back arching
        if "back_arch" in issues and score < 5.5:
            return False, "Excessive lower back arch"
        
        if len(issues) >= 3:
            return False, "Too many form issues"
        
        # ACCEPT CONDITIONS
        if score >= 5.5:  # Slightly stricter for press
            return True, "Rep counted"
        
        return False, "Form too poor"
    
    @staticmethod
    def should_count_situp(rep_score, feedback):
        """Sit-up rep validation"""
        issues = set(rep_score.get("invalid_reasons", []))
        score = rep_score.get("score", 10)
        no_data_keys = {"no_valid_data", "not_pushup_pattern", "no_body_alignment_data"}
        
        # REJECT CONDITIONS
        if issues.intersection(no_data_keys) or score < 4:
            return False, "Cannot verify form"
        
        # Incomplete range = not full rep
        if "limited_range" in issues or "incomplete_extension" in issues:
            return False, "Incomplete range of motion"
        
        # Neck pulling instead of core work
        if "neck_pull" in issues and score < 6:
            return False, "Using neck instead of core"
        
        if len(issues) >= 3:
            return False, "Too many form issues"
        
        # ACCEPT CONDITIONS
        if score >= 5:
            return True, "Rep counted"
        
        return False, "Form too poor"
    
    @staticmethod
    def should_count_mountain_climber(rep_score, feedback):
        """Mountain climber rep validation"""
        issues = set(rep_score.get("invalid_reasons", []))
        score = rep_score.get("score", 10)
        no_data_keys = {"no_valid_data", "not_pushup_pattern", "no_body_alignment_data"}
        
        # REJECT CONDITIONS
        if issues.intersection(no_data_keys) or score < 4:
            return False, "Cannot verify form"
        
        # Sagging hips = not in plank
        if "sagging_hips" in issues:
            return False, "Hips must stay level in plank"
        
        # Not actually doing mountain climbers
        if "slow_movement" in issues and score < 5:
            return False, "Movement too slow"
        
        if len(issues) >= 3:
            return False, "Too many form issues"
        
        # ACCEPT CONDITIONS
        if score >= 5:
            return True, "Rep counted"
        
        return False, "Form too poor"
    
    @staticmethod
    def validate_rep(exercise_key: str, rep_score: dict, feedback: list) -> tuple[bool, str]:
        """Route to appropriate exercise validator"""
        validators = {
            "pushup": ExerciseRepValidator.should_count_pushup,
            "squat": ExerciseRepValidator.should_count_squat,
            "lunge": ExerciseRepValidator.should_count_lunge,
            "bicep_curl": ExerciseRepValidator.should_count_bicep_curl,
            "shoulder_press": ExerciseRepValidator.should_count_shoulder_press,
            "situp": ExerciseRepValidator.should_count_situp,
            "mountain_climber": ExerciseRepValidator.should_count_mountain_climber,
        }
        
        validator = validators.get(exercise_key, ExerciseRepValidator.should_count_pushup)
        return validator(rep_score, feedback)


class ExerciseSafetyValidator:
    """Pre-rep checks for exercise readiness (like floor clearance for pushup)"""
    
    @staticmethod
    def is_pushup_ready(angles, floor_clearance):
        """Check if in ready pushup position"""
        if not angles or floor_clearance is None:
            return False
        
        hip_angle = angles.get('left_hip') or angles.get('right_hip')
        knee_angle = angles.get('left_knee') or angles.get('right_knee')
        
        if hip_angle is None or knee_angle is None:
            return False
        
        # Must be in plank position
        angle_ready = hip_angle >= 155 and knee_angle >= 165
        
        if not angle_ready:
            return False
        
        # Floor clearance checks
        if floor_clearance["knee_clearance_px"] < floor_clearance["knee_min_clearance_px"]:
            return False
        if floor_clearance["chest_clearance_px"] < floor_clearance["chest_min_clearance_px"]:
            return False
        
        return True
    
    @staticmethod
    def is_squat_ready(angles):
        """Check if in ready squat position - feet planted, upright"""
        if not angles:
            return False
        
        # Should be standing upright
        hip_angle = angles.get('left_hip') or angles.get('right_hip')
        knee_angle = angles.get('left_knee') or angles.get('right_knee')
        
        if hip_angle is None or knee_angle is None:
            return False
        
        # Standing position: hips and knees extended
        return hip_angle >= 160 and knee_angle >= 170
    
    @staticmethod
    def is_lunge_ready(angles):
        """Check if in ready lunge position"""
        if not angles:
            return False
        
        # Should be standing between lunges
        hip_angle = angles.get('left_hip') or angles.get('right_hip')
        knee_angle = angles.get('left_knee') or angles.get('right_knee')
        
        if hip_angle is None or knee_angle is None:
            return False
        
        return hip_angle >= 160 and knee_angle >= 170
    
    @staticmethod
    def is_bicep_curl_ready(angles):
        """Check if arms at starting position"""
        if not angles:
            return False
        
        elbow = angles.get('left_elbow') or angles.get('right_elbow')
        
        if elbow is None:
            return False
        
        # Arms should be extended or nearly extended
        return elbow >= 150
    
    @staticmethod
    def is_shoulder_press_ready(angles):
        """Check if at starting position (shoulders level)"""
        if not angles:
            return False
        
        return True  # Less strict, any standing position works
    
    @staticmethod
    def is_situp_ready(angles):
        """Check if lying flat at start"""
        if not angles:
            return False
        
        hip_angle = angles.get('left_hip') or angles.get('right_hip')
        
        if hip_angle is None:
            return False
        
        # Should be lying relatively flat
        return hip_angle <= 40
    
    @staticmethod
    def is_mountain_climber_ready(angles):
        """Check if in plank position"""
        if not angles:
            return False
        
        hip_angle = angles.get('left_hip') or angles.get('right_hip')
        
        if hip_angle is None:
            return False
        
        # Must be in plank
        return hip_angle >= 165
    
    @staticmethod
    def check_readiness(exercise_key: str, angles: dict, floor_clearance=None) -> bool:
        """Route to appropriate readiness check"""
        readiness_checks = {
            "pushup": lambda: ExerciseSafetyValidator.is_pushup_ready(angles, floor_clearance),
            "squat": lambda: ExerciseSafetyValidator.is_squat_ready(angles),
            "lunge": lambda: ExerciseSafetyValidator.is_lunge_ready(angles),
            "bicep_curl": lambda: ExerciseSafetyValidator.is_bicep_curl_ready(angles),
            "shoulder_press": lambda: ExerciseSafetyValidator.is_shoulder_press_ready(angles),
            "situp": lambda: ExerciseSafetyValidator.is_situp_ready(angles),
            "mountain_climber": lambda: ExerciseSafetyValidator.is_mountain_climber_ready(angles),
        }
        
        check = readiness_checks.get(exercise_key, lambda: True)
        return check()
