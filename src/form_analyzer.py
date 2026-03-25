"""
Enhanced Form Analyzer - Comprehensive per-rep form quality assessment
Focuses on injury prevention and proper technique
"""

from src.form_standards import assess_injury_risk, get_form_quality_score, categorize_form


class FormAnalyzer:
    """Track and analyze form for every repetition"""
    
    def __init__(self):
        self.rep_angles = []  # Collect angles throughout rep
        self.current_rep = 0
        self.head_readings = []  # Track head Y for calibration
        self.head_baseline = None  # Calibrated after 200 frames
        
    def collect(self, landmarks, angles):
        """Collect frame data + calibrate head position"""
        # Calibrate head baseline from first 200 frames
        if self.head_baseline is None and len(self.head_readings) < 200:
            if landmarks and len(landmarks) > 0:
                head_y = landmarks[0].get('y')
                if head_y:
                    self.head_readings.append(head_y)
                    if len(self.head_readings) >= 200:
                        self.head_baseline = sorted(self.head_readings)[100]  # Median
        
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
        """Check head alignment against baseline"""
        if self.head_baseline is None:
            return None
        
        head_y = self._get_head_y(bottom_landmarks)
        if head_y is None:
            return None
        
        # Calculate deviation as percentage of image height
        deviation = abs(head_y - self.head_baseline)
        deviation_pct = deviation * 100  # Normalize to % (assuming 0-1 range)
        
        if deviation_pct > 12:
            if head_y > self.head_baseline:
                return "⚠️ Head dropping - keep neutral"
            else:
                return "⚠️ Head position unstable - maintain neutral"
        
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
        positives = [f for f in risk_assessment['feedback'] if "✓" in f]
        if positives:
            for positive in positives:
                feedback.append(f"  ✓ {positive.replace('✓ ', '')}")
        
        # ─── HEAD POSITION CHECK ─────────────────────────
        head_check = self._check_head_position(bottom_landmarks)
        if head_check:
            feedback.append(f"  {head_check}")
        
        # ─── ISSUES ──────────────────────────────────────  
        if risk_assessment['flags']:
            for flag in risk_assessment['flags']:
                feedback.append(f"  🚨 {flag}")
        
        if risk_assessment['issues']:
            for issue in risk_assessment['issues']:
                feedback.append(f"  ⚠️ {issue.replace('_', ' ').title()}")
        
        # ─── CORRECTIONS ─────────────────────────────────
        corrections = [f for f in risk_assessment['feedback'] if "✓" not in f]
        if corrections:
            for correction in corrections:
                feedback.append(f"  💡 {correction}")
        
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
