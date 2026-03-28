"""
ROBUST EXERCISE VALIDATORS
Comprehensive form checking for each exercise with human variation tolerance
"""

import numpy as np
from .exercise_form_standards import (
    PushupFormStandards as PushupStd,
    SquatFormStandards as SquatStd,
    LungeFormStandards as LungeStd,
    BicepCurlFormStandards as CurlStd,
    ShoulderPressFormStandards as PressStd,
    SitupFormStandards as SitupStd,
    MountainClimberFormStandards as ClimbStd,
)


class PushupValidator:
    """Comprehensive push-up form validation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.elbows = []
        self.hips = []
        self.knees = []
        self.shoulders = []
        self.frame_angles = []
        
    def collect(self, landmarks, angles):
        if angles is None or landmarks is None:
            return
            
        elbow = angles.get('left_elbow') or angles.get('right_elbow')
        hip = angles.get('left_hip') or angles.get('right_hip')
        knee = angles.get('left_knee') or angles.get('right_knee')
        
        if elbow:
            self.elbows.append(elbow)
        if hip:
            self.hips.append(hip)
        if knee:
            self.knees.append(knee)
        
        self.frame_angles.append({'elbow': elbow, 'hip': hip, 'knee': knee})
    
    def evaluate(self):
        """Comprehensive push-up form evaluation"""
        if not self.elbows:
            return ["No valid data"]
        
        feedback = []
        issues = []
        score = 100
        
        # ─────── DEPTH ANALYSIS ──────────────
        min_elbow = np.percentile(self.elbows, 15)  # 15th percentile (bottom)
        max_elbow = np.percentile(self.elbows, 85)  # 85th percentile (top)
        avg_elbow = np.mean(self.elbows)
        
        if min_elbow > PushupStd.ELBOW_DEPTH_MAX:
            feedback.append("❌ Not going deep enough - aim for 75-90° at bottom")
            issues.append("insufficient_depth")
            score -= 25
        elif min_elbow < PushupStd.ELBOW_DEPTH_MIN:
            feedback.append("⚠️ Going too deep - risk of joint stress (stop at 75°)")
            issues.append("hyperextension")
            score -= 20
        else:
            feedback.append("✓ Good depth (75-95° at bottom)")
        
        # ─────── EXTENSION ANALYSIS ───────────
        if max_elbow < PushupStd.ELBOW_TOP_MIN:
            feedback.append("❌ Not fully extending arms at top (lock out elbows)")
            issues.append("no_lockout")
            score -= 20
        else:
            feedback.append("✓ Good arm extension at top")
        
        # ─────── HIP ALIGNMENT (SAGGING) ─────
        if self.hips:
            min_hip = np.percentile(self.hips, 20)
            avg_hip = np.mean(self.hips)
            
            if min_hip < PushupStd.HIP_MIN:
                feedback.append("🚨 Hips sagging - keep body straight (maintain 170°+)")
                issues.append("sagging_hips")
                score -= 25
            elif avg_hip > PushupStd.HIP_MAX:
                feedback.append("⚠️ Hips too high (pike position) - lower hips")
                issues.append("pike_position")
                score -= 15
            else:
                feedback.append("✓ Good body alignment (neutral spine)")
        
        # ─────── LEG EXTENSION ──────────────
        if self.knees:
            min_knee = np.percentile(self.knees, 25)
            
            if min_knee < PushupStd.KNEE_LENIENT_MIN:
                feedback.append("❌ Knees bending - keep legs straight (170°+)")
                issues.append("bent_knees")
                score -= 30
            else:
                feedback.append("✓ Legs fully extended")
        
        # ─────── CONSISTENCY CHECK ──────────
        depth_variance = max(self.elbows) - min(self.elbows)
        if depth_variance > 40:
            feedback.append("⚠️ Inconsistent depth - maintain steady range")
            issues.append("inconsistent_depth")
            score -= 10
        
        # ─────── FULL BODY ALIGNMENT ────────
        # Check if hips and knees both good
        if self.hips and self.knees:
            avg_hip = np.mean(self.hips)
            avg_knee = np.mean(self.knees)
            if avg_hip > 165 and avg_knee > 170:
                feedback.append("✓ Full body alignment: perfect plank position")
        
        score = max(0, min(100, score))
        is_valid = len(issues) < 3  # Allow minor issues
        
        return feedback


class SquatValidator:
    """Comprehensive squat form validation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.knees = []
        self.hips = []
        self.frame_data = []
    
    def collect(self, landmarks, angles):
        if angles is None:
            return
        
        knee = angles.get('left_knee') or angles.get('right_knee')
        hip = angles.get('left_hip') or angles.get('right_hip')
        
        if knee:
            self.knees.append(knee)
        if hip:
            self.hips.append(hip)
        
        self.frame_data.append({'knee': knee, 'hip': hip, 'landmarks': landmarks})
    
    def evaluate(self):
        """Comprehensive squat form evaluation"""
        if not self.knees:
            return ["No valid data"]
        
        feedback = []
        issues = []
        score = 100
        
        min_knee = np.percentile(self.knees, 20)
        max_knee = np.percentile(self.knees, 80)
        avg_knee = np.mean(self.knees)
        
        # ─────── DEPTH ANALYSIS ─────────────
        # Determine depth type achieved
        if min_knee < SquatStd.KNEE_DEPTH_ATG:
            feedback.append("✓ Full squat depth (below parallel) - excellent!")
            square_type = "full"
        elif min_knee < SquatStd.KNEE_DEPTH_PARALLEL:
            feedback.append("✓ Parallel squat (hip level with knees) - good")
            square_type = "parallel"
        elif min_knee < SquatStd.KNEE_DEPTH_FULL + 15:
            feedback.append("⚠️ Partial squat (approaching parallel) - go deeper")
            issues.append("shallow_depth")
            score -= 20
            square_type = "partial"
        else:
            feedback.append("❌ Too shallow - aim for at least parallel")
            issues.append("very_shallow_depth")
            score -= 30
            square_type = "very_shallow"
        
        # ─────── HIP POSITION ───────────────
        if self.hips:
            min_hip = np.percentile(self.hips, 20)
            
            # Hip angle relates to back angle
            if min_hip > 120:
                feedback.append("⚠️ Back angle too upright (good for knees, check ankle mobility)")
            elif min_hip < 60:
                feedback.append("⚠️ Hip crease too tight - may indicate limited mobility")
            else:
                feedback.append("✓ Good hip position")
        
        # ─────── HIP/KNEE COORDINATION ──────
        # Both should reach minimum at roughly the same time (good form)
        if self.knees and self.hips:
            min_knee_idx = np.argmin(self.knees)
            min_hip_idx = np.argmin(self.hips)
            timing_diff = abs(min_knee_idx - min_hip_idx)
            
            if timing_diff > 5:
                feedback.append("⚠️ Hip and knee descent uneven - coordinate timing")
                issues.append("uneven_descent")
                score -= 10
        
        # ─────── KNEE VALGUS CHECK ──────────
        # Check for knee cave (inward), would show as reduced knee angle variance
        range_variance = max(self.knees) - min(self.knees)
        if range_variance < 20:
            feedback.append("⚠️ Limited knee flexion - check for knee cave (knees caving in)")
            issues.append("limited_knee_motion")
            score -= 15
        
        # ─────── CONSISTENCY ────────────────
        cv = np.std(self.knees) / np.mean(self.knees) if np.mean(self.knees) > 0 else 0
        if cv > 0.15:  # 15% coefficient of variation
            feedback.append("⚠️ Inconsistent depth between reps - maintain steady depth")
            issues.append("inconsistent_depth")
            score -= 12
        
        # ─────── ASCENT CHECK ───────────────
        # Check hip and knee accelerate together on the way up
        if len(self.knees) > 10:
            ascent_knees = self.knees[-10:]  # Last 10 frames
            if all(ascent_knees[i] <= ascent_knees[i+1] for i in range(len(ascent_knees)-1)):
                feedback.append("✓ Controlled ascent")
            else:
                feedback.append("⚠️ Jerky ascent - move smoothly")
                score -= 8
        
        score = max(0, min(100, score))
        is_valid = len(issues) < 2
        
        return feedback


class LungeValidator:
    """Comprehensive lunge form validation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.front_knees = []
        self.back_knees = []
        self.hips = []
    
    def collect(self, landmarks, angles):
        if angles is None:
            return
        
        # For lunges, we'd need data from both legs
        knee_l = angles.get('left_knee')
        knee_r = angles.get('right_knee')
        hip = angles.get('left_hip') or angles.get('right_hip')
        
        # In a real scenario, we'd track which knee is front/back
        if knee_l:
            self.front_knees.append(knee_l)
        if knee_r:
            self.back_knees.append(knee_r)
        if hip:
            self.hips.append(hip)
    
    def evaluate(self):
        """Comprehensive lunge form evaluation"""
        if not self.front_knees:
            return ["No valid data"]
        
        feedback = []
        issues = []
        score = 100
        
        min_front = np.percentile(self.front_knees, 20)
        min_back = np.percentile(self.back_knees, 20) if self.back_knees else None
        
        # ─────── FRONT KNEE DEPTH ──────────
        if min_front < LungeStd.FRONT_KNEE_MIN:
            feedback.append("⚠️ Front lunge too shallow - lower hips more")
            issues.append("shallow_lunge")
            score -= 20
        elif min_front > LungeStd.FRONT_KNEE_MAX:
            feedback.append("⚠️ Front knee driven too far forward - risk of stress")
            issues.append("excessive_front_knee")
            score -= 15
        else:
            feedback.append("✓ Good front knee depth (60-90°)")
        
        # ─────── BACK KNEE ──────────────────
        if min_back and min_back < LungeStd.BACK_KNEE_MIN:
            feedback.append("⚠️ Back knee too high - lower it toward ground")
            issues.append("high_back_knee")
            score -= 15
        
        # ─────── TORSO POSITION ─────────────
        if self.hips:
            avg_hip = np.mean(self.hips)
            if avg_hip < LungeStd.TORSO_UPRIGHT_OPTIMAL - LungeStd.TORSO_FORWARD_LEAN_MAX:
                feedback.append("⚠️ Excessive forward lean - keep torso upright")
                issues.append("forward_lean")
                score -= 15
            else:
                feedback.append("✓ Good torso alignment")
        
        # ─────── BALANCE Check ──────────────
        # Front and back knee contraction should be coordinated
        if len(self.front_knees) > 5 and len(self.back_knees) > 5:
            front_std = np.std(self.front_knees)
            back_std = np.std(self.back_knees)
            if front_std > 30 or back_std > 30:
                feedback.append("⚠️ Unstable lunge - maintain balance")
                issues.append("unstable_lunge")
                score -= 15
        
        score = max(0, min(100, score))
        is_valid = len(issues) < 2
        
        return feedback


class BicepCurlValidator:
    """Comprehensive bicep curl form validation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.elbows = []
        self.frame_data = []
        self.speeds = []
    
    def collect(self, landmarks, angles):
        if angles is None:
            return
        
        elbow = angles.get('left_elbow') or angles.get('right_elbow')
        
        if elbow:
            self.elbows.append(elbow)
            # Track speed (change in angle)
            if len(self.elbows) > 1:
                self.speeds.append(abs(self.elbows[-1] - self.elbows[-2]))
        
        self.frame_data.append({'elbow': elbow, 'landmarks': landmarks})
    
    def evaluate(self):
        """Comprehensive curl form evaluation"""
        if not self.elbows:
            return ["No valid data"]
        
        feedback = []
        # For curls: max elbow angle is bottom extension, min elbow angle is top contraction.
        top_angle = float(np.percentile(self.elbows, 15))
        bottom_angle = float(np.percentile(self.elbows, 85))

        # ─────── BOTTOM EXTENSION ──────────
        if bottom_angle >= CurlStd.ELBOW_BOTTOM_LENIENT_MIN:
            feedback.append("Stable upper body")
        elif bottom_angle >= CurlStd.ELBOW_BOTTOM_MIN:
            feedback.append("Did not fully extend between reps")
        else:
            feedback.append("Did not fully extend between reps")

        # ─────── TOP CONTRACTION ───────────
        if 30 <= top_angle <= 60:
            feedback.append("Good curl contraction")
        elif top_angle < 30:
            feedback.append("Elbow collapsing too much")
        elif top_angle <= 90:
            feedback.append("Inconsistent curl depth")
        else:
            feedback.append("Curl range too small")

        # ─────── RANGE OF MOTION ───────────
        total_range = bottom_angle - top_angle
        if total_range < 65:
            feedback.append("Limited range of motion")

        # Momentum/consistency checks are intentionally omitted for curls to avoid
        # punishing normal tempo variation from camera jitter.

        return feedback


class ShoulderPressValidator:
    """Comprehensive shoulder press form validation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.elbows = []
        self.hips = []
        self.frame_data = []
    
    def collect(self, landmarks, angles):
        if angles is None:
            return
        
        elbow = angles.get('left_elbow') or angles.get('right_elbow')
        hip = angles.get('left_hip') or angles.get('right_hip')
        
        if elbow:
            self.elbows.append(elbow)
        if hip:
            self.hips.append(hip)
        
        self.frame_data.append({'elbow': elbow, 'hip': hip})
    
    def evaluate(self):
        """Comprehensive shoulder press evaluation"""
        if not self.elbows:
            return ["No valid data"]
        
        feedback = []
        issues = []
        score = 100
        
        min_elbow = np.percentile(self.elbows, 20)
        max_elbow = np.percentile(self.elbows, 85)
        
        # ─────── STARTING POSITION ─────────
        if min_elbow < PressStd.ELBOW_BOTTOM_MIN:
            feedback.append("⚠️ Starting position too low - bring elbows to shoulder height")
            issues.append("low_start")
            score -= 15
        elif min_elbow > PressStd.ELBOW_BOTTOM_MIN + 30:
            feedback.append("⚠️ Starting position too high - lower to shoulder level")
            issues.append("high_start")
            score -= 10
        else:
            feedback.append("✓ Good starting position")
        
        # ─────── LOCKOUT ────────────────────
        if max_elbow < PressStd.ELBOW_TOP_MIN:
            feedback.append("❌ Not locking out at top - extend fully overhead")
            issues.append("no_lockout")
            score -= 25
        elif max_elbow < PressStd.ELBOW_TOP_OPTIMAL:
            feedback.append("⚠️ Slight elbow bend at top - lock elbows")
            issues.append("partial_lockout")
            score -= 12
        else:
            feedback.append("✓ Full lockout achieved")
        
        # ─────── LOWER BACK COMPENSATION ───
        if self.hips:
            min_hip = np.percentile(self.hips, 20)
            
            if min_hip < PressStd.HIP_ANGLE_MIN:
                feedback.append("🚨 Excessive lower back arch - use core, not back")
                issues.append("back_arch")
                score -= 30
            else:
                feedback.append("✓ Good core stability (neutral spine)")
        
        # ─────── PRESS PATH ─────────────────
        # Check consistency of press (should be straight up)
        if len(self.elbows) > 10:
            press_phase = self.elbows[len(self.elbows)//3:2*len(self.elbows)//3]
            if press_phase:
                press_consistency = np.std(press_phase) / np.mean(press_phase) if np.mean(press_phase) > 0 else 0
                if press_consistency > 0.15:
                    feedback.append("⚠️ Inconsistent press path - press straight up")
                    issues.append("inconsistent_path")
                    score -= 15
        
        score = max(0, min(100, score))
        is_valid = len(issues) < 2
        
        return feedback


class SitupValidator:
    """Comprehensive sit-up form validation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.hips = []
        self.frame_data = []
    
    def collect(self, landmarks, angles):
        if angles is None:
            return
        
        hip = angles.get('left_hip') or angles.get('right_hip')
        
        if hip:
            self.hips.append(hip)
        
        self.frame_data.append({'hip': hip, 'landmarks': landmarks})
    
    def evaluate(self):
        """Comprehensive sit-up evaluation"""
        if not self.hips:
            return ["No valid data"]
        
        feedback = []
        issues = []
        score = 100
        
        min_hip = np.percentile(self.hips, 15)
        max_hip = np.percentile(self.hips, 85)
        total_range = max_hip - min_hip
        
        # ─────── RANGE OF MOTION ──────────
        if total_range < SitupStd.HIP_RANGE_MIN:
            feedback.append("⚠️ Limited range of motion - go through full ROM")
            issues.append("limited_range")
            score -= 25
        elif total_range < SitupStd.HIP_RANGE_OPTIMAL - 5:
            feedback.append("⚠️ Could increase range - fuller sit-ups better")
            issues.append("short_range")
            score -= 10
        else:
            feedback.append("✓ Good range of motion")
        
        # ─────── CONTRACTION DEPTH ─────────
        if max_hip < SitupStd.HIP_CONTRACTED_MIN:
            feedback.append("⚠️ Not contracting fully - sit higher")
            issues.append("partial_contraction")
            score -= 15
        else:
            feedback.append("✓ Good peak contraction")
        
        # ─────── STARTING POSITION ────────
        if min_hip > SitupStd.HIP_BOTTOM_THRESHOLD:
            feedback.append("⚠️ Not fully relaxing between reps - go back to start")
            issues.append("incomplete_extension")
            score -= 12
        
        # ─────── NECK POSITION ────────────
        # Check for neck pulling (would show as jerky motion at start)
        if len(self.hips) > 5:
            first_phase_speeds = []
            for i in range(min(5, len(self.hips)-1)):
                first_phase_speeds.append(abs(self.hips[i+1] - self.hips[i]))
            
            if first_phase_speeds and np.max(first_phase_speeds) > 5:
                feedback.append("⚠️ Jerky start - don't pull with neck, use core")
                issues.append("neck_pull")
                score -= 15
        
        # ─────── MOMENTUM ──────────────────
        if len(self.hips) > 10:
            # Check for sudden changes (momentum)
            speeds = [abs(self.hips[i+1] - self.hips[i]) for i in range(len(self.hips)-1)]
            if np.max(speeds) > 3:
                feedback.append("⚠️ Using momentum - control the movement")
                issues.append("momentum")
                score -= 12
        
        # ─────── CONSISTENCY ────────────────
        if len(self.hips) > 10:
            cv = np.std(self.hips) / np.mean(self.hips) if np.mean(self.hips) > 0 else 0
            if cv > 0.15:
                feedback.append("⚠️ Inconsistent form between reps - maintain steady tempo")
                issues.append("inconsistent")
                score -= 10
        
        score = max(0, min(100, score))
        is_valid = len(issues) < 2
        
        return feedback


class MountainClimberValidator:
    """Comprehensive mountain climber form validation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.front_knees = []
        self.back_knees = []
        self.hips = []
        self.frame_data = []
        self.rep_times = []
    
    def collect(self, landmarks, angles):
        if angles is None:
            return
        
        knee_l = angles.get('left_knee')
        knee_r = angles.get('right_knee')
        hip = angles.get('left_hip') or angles.get('right_hip')
        
        if knee_l:
            self.front_knees.append(knee_l)
        if knee_r:
            self.back_knees.append(knee_r)
        if hip:
            self.hips.append(hip)
        
        self.frame_data.append({'knee_l': knee_l, 'knee_r': knee_r, 'hip': hip})
    
    def evaluate(self):
        """Comprehensive mountain climber evaluation"""
        if not self.front_knees:
            return ["No valid data"]
        
        feedback = []
        issues = []
        score = 100
        
        # ─────── PLANK POSITION ─────────────
        if self.hips:
            hips = np.array(self.hips)
            min_hip = np.percentile(hips, 20)
            max_hip = np.percentile(hips, 80)
            avg_hip = np.mean(hips)
            
            if avg_hip < ClimbStd.HIP_SAGGING_THRESHOLD:
                feedback.append("🚨 Hips sagging - tighten core and keep body straight")
                issues.append("sagging_hips")
                score -= 30
            elif avg_hip > ClimbStd.HIP_PIKE_THRESHOLD:
                feedback.append("⚠️ Pike position - lower hips to neutral")
                issues.append("pike")
                score -= 15
            else:
                feedback.append("✓ Good plank position maintained")
            
            # ─────── HIP STABILITY ───────────
            hip_drift = max(hips) - min(hips)
            if hip_drift > 20:
                feedback.append("⚠️ Hip instability - keep hips level")
                issues.append("hip_drift")
                score -= 15
        
        # ─────── KNEE DRIVE ─────────────────
        if self.front_knees:
            min_knee = np.percentile(self.front_knees, 15)
            max_knee = np.percentile(self.front_knees, 85)
            
            # Front knee should drive high during contraction
            if max_knee < ClimbStd.KNEE_DRIVE_MIN:
                feedback.append("⚠️ Knee drive too low - drive knees toward chest")
                issues.append("low_knee_drive")
                score -= 20
            else:
                feedback.append("✓ Good knee drive height")
            
            # Back leg should extend fully
            if self.back_knees:
                back_min = np.percentile(self.back_knees, 20)
                if back_min < ClimbStd.KNEE_EXTENDED_MIN:
                    feedback.append("⚠️ Back leg not extending - straighten back leg")
                    issues.append("back_leg_bend")
                    score -= 15
        
        # ─────── CADENCE ────────────────────
        # Check if movement is rhythmic
        if len(self.front_knees) > 20:
            # Estimate cadence from oscillation
            first_half = self.front_knees[:len(self.front_knees)//2]
            oscillations = sum(1 for i in range(len(first_half)-1) if first_half[i] != first_half[i+1])
            
            if oscillations < 2:
                feedback.append("⚠️ Movement too slow or stuck - maintain climbing pace")
                issues.append("slow_movement")
                score -= 15
            else:
                feedback.append("✓ Good climbing rhythm")
        
        # ─────── CONSISTENCY ────────────────
        if self.front_knees and self.back_knees:
            front_cv = np.std(self.front_knees) / np.mean(self.front_knees) if np.mean(self.front_knees) > 0 else 0
            back_cv = np.std(self.back_knees) / np.mean(self.back_knees) if np.mean(self.back_knees) > 0 else 0
            
            if front_cv > 0.20 or back_cv > 0.20:
                feedback.append("⚠️ Inconsistent form - maintain steady position")
                issues.append("inconsistent")
                score -= 10
        
        score = max(0, min(100, score))
        is_valid = len(issues) < 2
        
        return feedback
