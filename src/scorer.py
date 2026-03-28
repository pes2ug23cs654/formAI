"""Scoring utilities for rep-level and session-level summaries."""

import builtins
from collections import Counter


CRITICAL_RULES = {
	"knees bent too much": "bent_knees",
	"no valid data": "not_pushup_pattern",
	"loss of balance": "unstable_form",
}

MAJOR_WARNING_RULES = {
	"too shallow": "shallow_depth",
	"too deep": "joint_stress",
	"hips sagging": "hip_sag",
	"hips too high": "pike_position",
	"too shallow squat": "shallow_squat",
	"too shallow lunge": "shallow_lunge",
	"sit-up range too shallow": "shallow_situp",
	"knee drive too shallow": "shallow_knee_drive",
	"curl range too small": "short_range_of_motion",
	"press depth too shallow": "short_range_of_motion",
	"knee angle too acute": "knee_overload",
	"elbow collapsing too much": "elbow_overload",
	"neck pulling likely": "neck_strain",
	"knees collapsing inward": "knee_valgus",
	"excessive forward lean": "forward_lean",
	"torso leaning too much": "forward_lean",
	"inconsistent squat depth": "inconsistent_depth",
	"inconsistent lunge depth": "inconsistent_depth",
	"inconsistent curl depth": "inconsistent_depth",
	"inconsistent press depth": "inconsistent_depth",
	"inconsistent sit-up range": "inconsistent_depth",
	"inconsistent knee drive": "inconsistent_depth",
	"lower back compensation": "core_instability",
	"body swinging too much": "momentum_cheat",
	"hip control unstable": "core_instability",
	"hips sagging during climbers": "hip_sag",
}

MINOR_WARNING_RULES = {
	"did not fully extend arms": "no_lockout",
	"head dropping": "head_drop",
	"hands too far": "wide_hands",
	"did not fully stand up": "no_lockout",
	"did not return to standing": "no_lockout",
	"did not return to start position": "no_lockout",
	"did not re-extend leg": "no_lockout",
	"did not lock out overhead": "no_lockout",
	"did not fully extend between reps": "no_lockout",
	"uneven hip alignment": "hip_asymmetry",
}

POSITIVE_MARKERS = {
	"good depth",
	"good body alignment",
	"good squat depth",
	"good lunge depth",
	"good torso control",
	"good hip alignment",
	"good curl contraction",
	"good press depth",
	"good sit-up range",
	"good knee drive",
	"stable upper body",
	"stable core alignment",
	"good trunk control",
	"good plank control",
}


def _normalize(message):
	"""Normalize any feedback value into a lowercase string safely."""
	try:
		if builtins.isinstance(message, builtins.str):
			return message.strip().lower()
		if message is None:
			return ""
		# Defensive fallback: handle booleans/numbers/other types without crashing.
		return builtins.str(message).strip().lower()
	except Exception:
		# Last-resort guard: never let feedback normalization crash analysis.
		return ""


def _iter_feedback_messages(feedback_messages):
	"""Yield flat string messages from possibly nested feedback structures."""
	for message in feedback_messages or []:
		if builtins.isinstance(message, (list, tuple, set)):
			for nested in message:
				yield nested
		else:
			yield message


def classify_feedback_message(message):
	"""Map feedback text to issue key and severity when possible."""
	msg = _normalize(message)

	for marker, issue_key in CRITICAL_RULES.items():
		if marker in msg:
			return issue_key, "critical"

	for marker, issue_key in MAJOR_WARNING_RULES.items():
		if marker in msg:
			return issue_key, "major_warning"

	for marker, issue_key in MINOR_WARNING_RULES.items():
		if marker in msg:
			return issue_key, "minor_warning"

	for marker in POSITIVE_MARKERS:
		if marker in msg:
			return None, "positive"

	return None, "neutral"


def score_rep(feedback_messages):
	"""Compute lenient score and validity from rep feedback strings."""
	score = 100
	issues = []
	warnings = []
	critical_count = 0
	major_warning_count = 0
	invalid_reasons = []

	for message in _iter_feedback_messages(feedback_messages):
		issue_key, severity = classify_feedback_message(message)

		if severity == "critical":
			score -= 35
			critical_count += 1
			if issue_key:
				issues.append(issue_key)
				invalid_reasons.append(issue_key)
		elif severity == "major_warning":
			score -= 12
			major_warning_count += 1
			if issue_key:
				issues.append(issue_key)
				warnings.append(issue_key)
		elif severity == "minor_warning":
			score -= 6
			if issue_key:
				issues.append(issue_key)
				warnings.append(issue_key)

	score = max(0, min(100, score))

	# Invalidate only when form is severely off, not for normal human variation.
	form_breakdown = major_warning_count >= 3 and score < 55
	if form_breakdown:
		invalid_reasons.append("form_breakdown")

	is_valid = (critical_count == 0) and (not form_breakdown)

	return {
		"score": score,
		"is_valid": is_valid,
		"issues": sorted(set(issues)),
		"warnings": sorted(set(warnings)),
		"invalid_reasons": sorted(set(invalid_reasons)),
	}


def summarize_session(rep_reports):
	"""Aggregate rep reports into one session-level summary."""
	if not rep_reports:
		return {
			"avg_score": 0,
			"valid_reps": 0,
			"invalid_reps": 0,
			"issue_counts": {},
			"top_issues": [],
		}

	total = len(rep_reports)
	valid_reps = sum(1 for rep in rep_reports if rep.get("is_valid"))
	invalid_reps = total - valid_reps
	avg_score = round(sum(rep.get("score", 0) for rep in rep_reports) / total, 1)

	issue_counter = Counter()
	for rep in rep_reports:
		issue_counter.update(rep.get("issues", []))

	return {
		"avg_score": avg_score,
		"valid_reps": valid_reps,
		"invalid_reps": invalid_reps,
		"issue_counts": dict(issue_counter),
		"top_issues": [k for k, _ in issue_counter.most_common(3)],
	}
