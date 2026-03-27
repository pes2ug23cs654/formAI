import numpy as np

# MediaPipe landmark indices
LANDMARKS = {
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13,    'right_elbow': 14,
    'left_wrist': 15,    'right_wrist': 16,
    'left_hip': 23,      'right_hip': 24,
    'left_knee': 25,     'right_knee': 26,
    'left_ankle': 27,    'right_ankle': 28,
}


# ── Angle calculation (3D-aware) ─────────────────────────
def calculate_angle(landmarks, a, b, c):
    """
    Calculate angle between three joints using 3D coordinates.
    This eliminates camera-angle bias by working in true 3D space.
    """
    try:
        # Try 3D calculation first (if z-coordinates available)
        lm_a = landmarks[a]
        lm_b = landmarks[b]
        lm_c = landmarks[c]
        
        if 'z' in lm_a and 'z' in lm_b and 'z' in lm_c:
            # 3D angle calculation (camera-angle independent)
            # Scale z appropriately: z is [-1,1] normalized, x/y are pixels
            # Multiply by 100 to give z relative weight vs pixel coordinates
            p1 = np.array([lm_a['x'], lm_a['y'], lm_a['z'] * 100])
            p2 = np.array([lm_b['x'], lm_b['y'], lm_b['z'] * 100])
            p3 = np.array([lm_c['x'], lm_c['y'], lm_c['z'] * 100])
        else:
            # Fallback to 2D if z not available
            p1 = np.array([lm_a['x'], lm_a['y']])
            p2 = np.array([lm_b['x'], lm_b['y']])
            p3 = np.array([lm_c['x'], lm_c['y']])
        
        v1 = p1 - p2
        v2 = p3 - p2

        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

        return round(angle, 1)

    except:
        return None


# ── Get all joint angles ─────────────────────
def get_all_angles(landmarks):
    if landmarks is None:
        return {}

    L = LANDMARKS

    angles = {}

    # Elbows
    angles['left_elbow'] = calculate_angle(
        landmarks, L['left_shoulder'], L['left_elbow'], L['left_wrist']
    )

    angles['right_elbow'] = calculate_angle(
        landmarks, L['right_shoulder'], L['right_elbow'], L['right_wrist']
    )

    # Hips (body alignment)
    angles['left_hip'] = calculate_angle(
        landmarks, L['left_shoulder'], L['left_hip'], L['left_knee']
    )

    angles['right_hip'] = calculate_angle(
        landmarks, L['right_shoulder'], L['right_hip'], L['right_knee']
    )

    return angles


# ── Visibility check ─────────────────────────
def get_visibility(landmarks, indices):
    if landmarks is None:
        return False

    return all(
        landmarks.get(i, {}).get('visibility', 0) > 0.5
        for i in indices
    )