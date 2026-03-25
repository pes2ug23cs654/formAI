import cv2
import mediapipe as mp


class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        return results

    def get_landmarks(self, results, shape):
        if not results.pose_landmarks:
            return None

        h, w = shape[:2]
        landmarks = {}

        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[i] = {
                "x": int(lm.x * w),
                "y": int(lm.y * h),
                "visibility": lm.visibility
            }

        return landmarks

    def draw_skeleton(self, frame, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame