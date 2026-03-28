"""
Session Analyzer - Aggregates all reps and generates consolidated feedback
Provides overall session score and AI Coach recommendations based on patterns
"""

from collections import Counter


class SessionAnalyzer:
    """Analyze complete workout session across all repetitions"""
    
    def __init__(self):
        self.all_reps = []  # Store each rep's analysis
    
    def add_rep(self, rep_data):
        """Add a completed rep's analysis"""
        self.all_reps.append(rep_data)
    
    def analyze(self):
        """
        Generate consolidated session report:
        - Overall quality score
        - Primary form issues (by frequency)
        - Targeted coach feedback
        """
        if not self.all_reps:
            return {
                "total_reps": 0,
                "overall_score": 0,
                "feedback": ["No reps completed"]
            }
        
        total_reps = len(self.all_reps)
        overall_score = sum(r.get('score', 0) for r in self.all_reps) // total_reps
        
        # Extract all issues and their severity
        all_issues = []
        severity_flags = Counter()
        
        for rep_idx, rep in enumerate(self.all_reps, 1):
            issues = rep.get('issues', [])
            flags = rep.get('flags', [])
            
            for issue in issues:
                all_issues.append({
                    'rep': rep_idx,
                    'issue': issue,
                    'score': rep.get('score', 0)
                })
            
            for flag in flags:
                severity_flags[flag] += 1
        
        # Count issue frequency
        issue_counts = Counter(i['issue'] for i in all_issues)
        
        # Build prioritized coach feedback
        feedback = []
        
        # Header
        feedback.append("=" * 60)
        feedback.append(f"SESSION SUMMARY - {total_reps} Reps Completed")
        feedback.append(f"Overall Quality: {overall_score}/100")
        feedback.append("=" * 60)
        
        # Identify patterns
        if issue_counts:
            feedback.append("\n🎯 PRIMARY FORM ISSUES (by frequency):")
            
            for issue, count in issue_counts.most_common(3):
                percentage = round(100 * count / total_reps)
                if percentage >= 66:  # Present in 2/3+ reps
                    feedback.append(f"  • {issue}: {count}/{total_reps} reps ({percentage}%)")
        
        # Severity analysis
        if severity_flags['CRITICAL'] > 0:
            feedback.append(f"\n⚠️  CRITICAL ISSUES: {severity_flags['CRITICAL']} instances")
            feedback.append("  Action: Modify form immediately to prevent injury")
        elif severity_flags['MODERATE'] > 0:
            feedback.append(f"\n⚡ MODERATE ISSUES: {severity_flags['MODERATE']} instances")
            feedback.append("  Action: Focus on these form cues in next session")
        
        # Trend analysis
        scores = [r.get('score', 0) for r in self.all_reps]
        if len(scores) >= 2:
            trend = "improving 📈" if scores[-1] > scores[0] else ("consistent ➡️" if abs(scores[-1] - scores[0]) <= 5 else "declining 📉")
            feedback.append(f"\n📊 TREND: {trend} ({scores[0]}/100 → {scores[-1]}/100)")
        
        # Generate targeted tips
        tips = self._generate_targeted_tips(issue_counts, all_issues)
        if tips:
            feedback.append("\n💡 COACH TIPS FOR NEXT SESSION:")
            for tip in tips:
                feedback.append(f"  → {tip}")
        
        feedback.append("\n" + "=" * 60)
        
        return {
            "total_reps": total_reps,
            "overall_score": overall_score,
            "feedback": feedback,
            "primary_issues": list(issue_counts.most_common(3)),
            "trend": self._get_trend(scores) if len(scores) >= 2 else "N/A"
        }
    
    def _generate_targeted_tips(self, issue_counts, all_issues):
        """Generate prioritized coaching tips based on actual issues"""
        tips = []
        
        if issue_counts.get('excessive_depth', 0) >= 1:
            reps_with_issue = [i['rep'] for i in all_issues if i['issue'] == 'excessive_depth']
            tips.append(f"DEPTH: Stop at 75-80° elbows. You went too deep in reps {reps_with_issue}. "
                       "Protect shoulders by avoiding excessive compression.")
        
        if issue_counts.get('sagging_hips', 0) >= 1 or issue_counts.get('mild_sagging', 0) >= 1:
            reps_with_issue = [i['rep'] for i in all_issues if i['issue'] in ['sagging_hips', 'mild_sagging']]
            tips.append(f"HIPS: Tighten core and keep 170-185° hip angle. "
                       f"Sagging detected in reps {reps_with_issue}. Engage glutes!")
        
        if issue_counts.get('pike_position', 0) >= 1:
            tips.append("PIKE: Avoid pike position (hips too high). Keep hips aligned with shoulders. "
                       "Maintain neutral spine throughout the movement.")
        
        if issue_counts.get('hyperextended_elbow', 0) >= 1:
            tips.append("ELBOWS: Don't lock/hyperextend at top. Stop at ~160°. "
                       "Prevents joint damage and maintains muscle tension.")
        
        if issue_counts.get('insufficient_depth', 0) >= 1:
            tips.append("DEPTH: Go deeper! Aim for 90° elbows at the bottom. "
                       "Shallow reps rob you of full chest activation.")
        
        if issue_counts.get('bent_knees', 0) >= 1:
            tips.append("LEGS: Keep legs straight and locked throughout. "
                       "Bent knees reduce core engagement. Full body tension required!")
        
        return tips[:3]  # Return top 3 tips
    
    def _get_trend(self, scores):
        """Analyze improvement trend"""
        if not scores or len(scores) < 2:
            return "insufficient_data"
        
        start = scores[0]
        end = scores[-1]
        
        if end > start + 5:
            return "improving"
        elif end < start - 5:
            return "declining"
        else:
            return "consistent"
