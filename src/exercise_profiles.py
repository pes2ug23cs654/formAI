"""Exercise profiles and analyzers for multi-exercise support."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.rep_analyzer import RepAnalyzer
from src.exercise_validators import (
    PushupValidator,
    SquatValidator,
    LungeValidator,
    BicepCurlValidator,
    ShoulderPressValidator,
    SitupValidator,
    MountainClimberValidator,
)


@dataclass(frozen=True)
class ExerciseProfile:
    key: str
    display_name: str
    primary_angle: str
    default_descent_trigger: float
    default_ascent_threshold: float
    calibration_descent_bounds: tuple[float, float]
    calibration_ascent_bounds: tuple[float, float]


class AngleWindowAnalyzer:
    """Reusable analyzer for exercises defined by a primary angle window."""

    def __init__(self, primary_key, good_low, good_high, shallow_cutoff, overdeep_cutoff, labels):
        self.primary_key = primary_key
        self.good_low = good_low
        self.good_high = good_high
        self.shallow_cutoff = shallow_cutoff
        self.overdeep_cutoff = overdeep_cutoff
        self.labels = labels
        self.reset()

    def reset(self):
        self.primary = []
        self.secondary = []

    def collect(self, landmarks, angles):
        if angles is None:
            return

        primary_val = angles.get(f"left_{self.primary_key}") or angles.get(f"right_{self.primary_key}")
        if primary_val is not None:
            self.primary.append(primary_val)

        hip = angles.get("left_hip") or angles.get("right_hip")
        if hip is not None:
            self.secondary.append(hip)

    def evaluate(self):
        if not self.primary:
            return ["No valid data"]

        feedback = []
        min_primary = float(np.percentile(self.primary, 10))
        max_primary = float(np.percentile(self.primary, 90))

        if min_primary > self.shallow_cutoff:
            feedback.append(self.labels["shallow"])
        elif min_primary < self.overdeep_cutoff:
            feedback.append(self.labels["overdeep"])
        elif self.good_low <= min_primary <= self.good_high:
            feedback.append(self.labels["good_depth"])
        else:
            feedback.append(self.labels["depth_warn"])

        if max_primary < self.labels.get("stand_threshold", 160):
            feedback.append(self.labels["lockout_warn"])

        if self.secondary:
            min_hip = float(np.percentile(self.secondary, 15))
            if min_hip < self.labels.get("hip_low", 50):
                feedback.append(self.labels["hip_warn"])
            else:
                feedback.append(self.labels["hip_good"])

        return feedback


class SquatAnalyzer(AngleWindowAnalyzer):
    def __init__(self):
        super().__init__(
            primary_key="knee",
            good_low=65,
            good_high=105,
            shallow_cutoff=125,
            overdeep_cutoff=55,
            labels={
                "shallow": "Too shallow squat",
                "overdeep": "Knee angle too acute",
                "good_depth": "Good squat depth",
                "depth_warn": "Inconsistent squat depth",
                "lockout_warn": "Did not fully stand up",
                "stand_threshold": 165,
                "hip_warn": "Excessive forward lean",
                "hip_low": 55,
                "hip_good": "Good torso control",
            },
        )


class LungeAnalyzer(AngleWindowAnalyzer):
    def __init__(self):
        super().__init__(
            primary_key="knee",
            good_low=70,
            good_high=110,
            shallow_cutoff=120,
            overdeep_cutoff=60,
            labels={
                "shallow": "Too shallow lunge",
                "overdeep": "Knee angle too acute",
                "good_depth": "Good lunge depth",
                "depth_warn": "Inconsistent lunge depth",
                "lockout_warn": "Did not return to standing",
                "stand_threshold": 160,
                "hip_warn": "Torso leaning too much",
                "hip_low": 50,
                "hip_good": "Good hip alignment",
            },
        )


class BicepCurlAnalyzer(AngleWindowAnalyzer):
    def __init__(self):
        super().__init__(
            primary_key="elbow",
            good_low=35,
            good_high=75,
            shallow_cutoff=95,
            overdeep_cutoff=25,
            labels={
                "shallow": "Curl range too small",
                "overdeep": "Elbow collapsing too much",
                "good_depth": "Good curl contraction",
                "depth_warn": "Inconsistent curl depth",
                "lockout_warn": "Did not fully extend between reps",
                "stand_threshold": 150,
                "hip_warn": "Body swinging too much",
                "hip_low": 45,
                "hip_good": "Stable upper body",
            },
        )


class ShoulderPressAnalyzer(AngleWindowAnalyzer):
    def __init__(self):
        super().__init__(
            primary_key="elbow",
            good_low=70,
            good_high=110,
            shallow_cutoff=125,
            overdeep_cutoff=50,
            labels={
                "shallow": "Press depth too shallow",
                "overdeep": "Elbows tucked excessively",
                "good_depth": "Good press depth",
                "depth_warn": "Inconsistent press depth",
                "lockout_warn": "Did not lock out overhead",
                "stand_threshold": 155,
                "hip_warn": "Lower back compensation",
                "hip_low": 45,
                "hip_good": "Stable core alignment",
            },
        )


class SitupAnalyzer(AngleWindowAnalyzer):
    def __init__(self):
        super().__init__(
            primary_key="hip",
            good_low=45,
            good_high=95,
            shallow_cutoff=120,
            overdeep_cutoff=30,
            labels={
                "shallow": "Sit-up range too shallow",
                "overdeep": "Neck pulling likely",
                "good_depth": "Good sit-up range",
                "depth_warn": "Inconsistent sit-up range",
                "lockout_warn": "Did not return to start position",
                "stand_threshold": 150,
                "hip_warn": "Hip control unstable",
                "hip_low": 40,
                "hip_good": "Good trunk control",
            },
        )


class MountainClimberAnalyzer(AngleWindowAnalyzer):
    def __init__(self):
        super().__init__(
            primary_key="knee",
            good_low=55,
            good_high=95,
            shallow_cutoff=120,
            overdeep_cutoff=40,
            labels={
                "shallow": "Knee drive too shallow",
                "overdeep": "Knee angle too acute",
                "good_depth": "Good knee drive",
                "depth_warn": "Inconsistent knee drive",
                "lockout_warn": "Did not re-extend leg",
                "stand_threshold": 160,
                "hip_warn": "Hips sagging during climbers",
                "hip_low": 150,
                "hip_good": "Good plank control",
            },
        )


EXERCISE_PROFILES = {
    "pushup": ExerciseProfile(
        key="pushup",
        display_name="Push-up",
        primary_angle="elbow",
        default_descent_trigger=110.0,
        default_ascent_threshold=155.0,
        calibration_descent_bounds=(95.0, 130.0),
        calibration_ascent_bounds=(145.0, 170.0),
    ),
    "squat": ExerciseProfile(
        key="squat",
        display_name="Squat",
        primary_angle="knee",
        default_descent_trigger=130.0,
        default_ascent_threshold=165.0,
        calibration_descent_bounds=(115.0, 145.0),
        calibration_ascent_bounds=(155.0, 175.0),
    ),
    "lunge": ExerciseProfile(
        key="lunge",
        display_name="Lunge",
        primary_angle="knee",
        default_descent_trigger=125.0,
        default_ascent_threshold=162.0,
        calibration_descent_bounds=(110.0, 140.0),
        calibration_ascent_bounds=(150.0, 172.0),
    ),
    "bicep_curl": ExerciseProfile(
        key="bicep_curl",
        display_name="Bicep Curl",
        primary_angle="elbow",
        default_descent_trigger=105.0,
        default_ascent_threshold=150.0,
        calibration_descent_bounds=(90.0, 125.0),
        calibration_ascent_bounds=(140.0, 170.0),
    ),
    "shoulder_press": ExerciseProfile(
        key="shoulder_press",
        display_name="Shoulder Press",
        primary_angle="elbow",
        default_descent_trigger=120.0,
        default_ascent_threshold=155.0,
        calibration_descent_bounds=(105.0, 135.0),
        calibration_ascent_bounds=(145.0, 170.0),
    ),
    "situp": ExerciseProfile(
        key="situp",
        display_name="Sit-up",
        primary_angle="hip",
        default_descent_trigger=115.0,
        default_ascent_threshold=140.0,
        calibration_descent_bounds=(90.0, 130.0),
        calibration_ascent_bounds=(120.0, 160.0),
    ),
    "mountain_climber": ExerciseProfile(
        key="mountain_climber",
        display_name="Mountain Climber",
        primary_angle="knee",
        default_descent_trigger=120.0,
        default_ascent_threshold=160.0,
        calibration_descent_bounds=(105.0, 140.0),
        calibration_ascent_bounds=(145.0, 172.0),
    ),
}


def get_exercise_profile(exercise_key: str) -> ExerciseProfile:
    key = (exercise_key or "pushup").lower()
    if key not in EXERCISE_PROFILES:
        raise ValueError(f"Unsupported exercise: {exercise_key}")
    return EXERCISE_PROFILES[key]


def list_exercise_keys() -> list[str]:
    return sorted(EXERCISE_PROFILES.keys())


def build_analyzer(exercise_key: str):
    key = (exercise_key or "pushup").lower()
    analyzers: dict[str, Callable[[], object]] = {
        "pushup": PushupValidator,
        "squat": SquatValidator,
        "lunge": LungeValidator,
        "bicep_curl": BicepCurlValidator,
        "shoulder_press": ShoulderPressValidator,
        "situp": SitupValidator,
        "mountain_climber": MountainClimberValidator,
    }
    return analyzers[key]()