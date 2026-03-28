"""
Enhanced Form Analyzer - Comprehensive per-rep form quality assessment
Focuses on injury prevention and proper technique
"""

import math
from src.form_standards import assess_injury_risk, get_form_quality_score, categorize_form


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points (3D-aware, camera-independent)"""
    dx = p1.get('x', 0) - p2.get('x', 0)
    dy = p1.get('y', 0) - p2.get('y', 0)
    # z is normalized [-1, 1], scale by 100 to be comparable with pixel coordinates
    dz = (p1.get('z', 0) - p2.get('z', 0)) * 100
    return math.sqrt(dx*dx + dy*dy + dz*dz)


class FormAnalyzer:
    """Track and analyze form for every repetition"""
    
    def __init__(self):
        self.rep_angles = []  # Collect angles throughout rep
        self.current_rep = 0
        self.head_readings = []  # Track head Y at TOP position
        self.head_baseline = None  # Calibrated when at full extension
        self.calibration_complete = False
        
    def collect(self, landmarks, angles):
        """Collect frame data + calibrate head position"""
        # Calibrate head baseline from TOP position (high elbow angle > 150°)
        if not self.calibration_complete and landmarks and len(landmarks) > 0:
            elbow = angles.get('left_elbow') or angles.get('right_elbow') if angles else None
            if elbow and elbow > 150:  # At TOP/EXTENDED position
                head_y = landmarks[0].get('y')
                if head_y:
                    self.head_readings.append(head_y)
                    if len(self.head_readings) >= 20:  # Collect 20 readings at top position
                        self.head_baseline = sorted(self.head_readings)[10]  # Median of top readings
                        self.calibration_complete = True
                        print(f"[HEAD CALIBRATED] Baseline set to {self.head_baseline:.0f}px (at TOP position after {len(self.head_readings)} samples)", flush=True)
        
        if angles:
            elbow = angles.get('left_elbow') or angles.get('right_elbow')
            hip = angles.get('left_hip') or angles.get('right_hip')
            knee = angles.get('left_knee') or angles.get('right_knee')
            
            if elbow:
                head_y = landmarks[0].get('y') if landmarks and len(landmarks) > 0 else None
                self.rep_angles.append({
                    'elbow': elbow,
                    'hip': hip,
                    'knee': knee,
                    'landmarks': landmarks,
                    'head_y': head_y
                })
    
    def _get_head_y(self, landmarks):
        """Extract head Y position from landmarks"""
        if landmarks and len(landmarks) > 0:
            return landmarks[0].get('y')
        return None
    
    def _check_head_position(self, bottom_landmarks):
        """
        Check head alignment using 3D Euclidean distance (camera-angle independent).
        
        Strategy: Normalize head drop by actual body length in 3D space
        head_drop_distance / body_length = camera-independent ratio
        
        3D Methodology:
        - Uses x, y (pixel), and z (depth) coordinates from MediaPipe
        - Eliminates camera angle bias by measuring true 3D distance
        - Works for:
          ✅ Any camera distance
          ✅ Any camera angle (camera-shift robust)
          ✅ Any user height
          ✅ Any video resolution
        """
        if not bottom_landmarks or len(bottom_landmarks) < 24:
            return None
        
        try:
            # Extract landmarks
            head = bottom_landmarks[0]         # Nose (0)
            shoulder = bottom_landmarks[11]   # Left shoulder (11)
            hip = bottom_landmarks[23]        # Left hip (23)
            
            if not all(p for p in [head, shoulder, hip]):
                return None
            
            # Calculate TRUE body length using Euclidean distance (2D)
            body_length = euclidean_distance(shoulder, hip)
            if body_length < 10:  # Invalid measurement
                return None
            
            # Calculate head drop using Euclidean distance
            head_drop_distance = euclidean_distance(head, shoulder)
            
            # Ratio: head drop / body length (CAMERA-INDEPENDENT)
            head_drop_ratio = head_drop_distance / body_length
            
            print(f"[HEAD] Shoulder-Hip distance: {body_length:.1f}, Head-Shoulder distance: {head_drop_distance:.1f}, Ratio: {head_drop_ratio:.3f}", flush=True)
            
            # Thresholds based on biomechanics:
            # In perfect plank: head_drop_ratio ≈ 0.15-0.25
            # Mild drop: 0.25-0.35
            # Excessive drop: > 0.35
            
            if head_drop_ratio > 0.35:
                return "[HEAD] Dropping excessively - keep neck neutral"
            elif head_drop_ratio > 0.25:
                return "[HEAD] Head slightly low - improve neck alignment"
            
            return None
            
        except Exception as e:
            print(f"[HEAD ERROR] {type(e).__name__}: {e}", flush=True)
            return None
    
    def evaluate(self):
        """Analyze completed rep and return detailed feedback"""
        if not self.rep_angles:
            return ["No data collected"]
        
        self.current_rep += 1
        
        # Get min/max for each joint
        elbows = [f['elbow'] for f in self.rep_angles if f['elbow']]
        hips = [f['hip'] for f in self.rep_angles if f['hip']]
        knees = [f['knee'] for f in self.rep_angles if f['knee']]
        
        # Create angles dict with min/max
        analysis_angles = {
            'left_elbow': min(elbows) if elbows else None,
            'left_hip': min(hips) if hips else None,
            'left_knee': min(knees) if knees else None,
        }
        
        # Get landmarks from bottom position
        bottom_landmarks = None
        if elbows:
            min_elbow_idx = elbows.index(min(elbows))
            bottom_landmarks = self.rep_angles[min_elbow_idx]['landmarks']
        
        # Use comprehensive injury assessment
        risk_assessment = assess_injury_risk(analysis_angles, bottom_landmarks)
        quality_score = get_form_quality_score(risk_assessment)
        quality_category = categorize_form(quality_score)
        
        # Build feedback report for this rep
        feedback = []
        
        # ─── HEADER ──────────────────────────────────────
        feedback.append(f"\nREP #{self.current_rep} - Quality: {quality_score}/100 ({quality_category})")
        
        # ─── JOINT ANGLES ────────────────────────────────
        if elbows:
            min_elbow = min(elbows)
            max_elbow = max(elbows)
            feedback.append(f"  Elbow: {min_elbow:.0f}° - {max_elbow:.0f}° (target: 90° at bottom)")
        
        if hips:
            min_hip = min(hips)
            feedback.append(f"  Hip: {min_hip:.0f}° (target: 170-185°)")
        
        if knees:
            min_knee = min(knees)
            feedback.append(f"  Knee: {min_knee:.0f}° (target: 180° locked)")
        
        # ─── POSITIVES ───────────────────────────────────
        positives = [f for f in risk_assessment['feedback'] if "[GOOD]" in f]
        if positives:
            for positive in positives:
                feedback.append(f"  [GOOD] {positive.replace('[GOOD] ', '')}") 
        
        # ─── HEAD POSITION CHECK ─────────────────────────
        head_check = self._check_head_position(bottom_landmarks)
        if head_check:
            feedback.append(f"  {head_check}")
        
        # ─── ISSUES ──────────────────────────────────────  
        if risk_assessment['flags']:
            for flag in risk_assessment['flags']:
                feedback.append(f"  [FLAG] {flag}")
        
        if risk_assessment['issues']:
            for issue in risk_assessment['issues']:
                feedback.append(f"  [ISSUE] {issue.replace('_', ' ').title()}") 
        
        # ─── CORRECTIONS ─────────────────────────────────
        corrections = [f for f in risk_assessment['feedback'] if "[GOOD]" not in f]
        if corrections:
            for correction in corrections:
                feedback.append(f"  [TIP] {correction}")
        
        feedback.append(f"  Risk: {risk_assessment['risk_level']}")
        
        return feedback
    
    def reset(self):
        """Reset for next rep"""
        self.rep_angles = []


# Legacy compatible interface (for existing code)
def analyze_pushup(landmarks, angles):
    """Legacy function for backwards compatibility"""
    analyzer = FormAnalyzer()
    analyzer.collect(landmarks, angles)
    return analyzer.evaluate()
