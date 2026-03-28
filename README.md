# Push-up Form Coach - AI-Powered Analysis System

## Quick Start

### 1. Activate Virtual Environment

```bash
# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies (if not already installed)

```bash
pip install -r requirements.txt
```

### 3. Run on a Video (New CLI)

```bash
python app.py --input <input_video.mp4> --output <output_video.mp4> --report-json <report.json>
```

Supported exercises:

- `pushup`
- `squat`
- `lunge`
- `bicep_curl`
- `shoulder_press`
- `situp`
- `mountain_climber`

**Example:**

```bash
python app.py --input input/input.mp4 --output output/output.mp4 --report-json report/report.json
python app.py --input input/long.mp4 --output output/output_long.mp4 --report-json report/report_long.json
python app.py --exercise squat --input input/input.mp4 --output output/output_squat.mp4 --report-json report/report_squat.json
python app.py --exercise bicep_curl --input input/input.mp4 --output output/output_curl.mp4 --report-json report/report_curl.json

# Optional tuning
python app.py --input input/input.mp4 --output output/output.mp4 --report-json report/report.json --calibration-seconds 10 --confidence-threshold 0.55
```

You can still run the legacy script directly:

```bash
python video_processor.py input/input.mp4
```

### 4. What to Expect

**Output:**

- Console output with rep counts and form feedback
- Generated video file with skeleton overlay
- Real-time angle measurements and state transitions
- Structured JSON report with rep-level scoring and validity
- Calibration metadata (dynamic thresholds)
- Confidence gating metadata (low-confidence frame tracking)

**Short video fallback:**

- If video length is shorter than calibration duration, FormAI auto-falls back to reduced calibration frames.
- If still not enough calibration samples are available, it safely uses default thresholds.

**Sample Output:**

```
[FRAME   50] Angle:  147.3° | State: DOWN

*** REP 1 ***
  - ✓ Elbow depth: Good (90° rule)
  - ✓ Hip alignment: Good (neutral spine)
  - Risk: OK

[FRAME  100] Angle:   55.6° | State: DOWN

*** REP 2 ***
  - Elbow: 85° - 175° (target: 90° at bottom)
  - Hip: 172° (target: 170-185°)
  - Knee: 180° (target: 180° locked)
  - ✓ Elbow depth: Good
  - Risk: OK

==================================================
Total reps: 4
==================================================
```

### 5. Frontend (Streamlit)

Run the frontend app:

```bash
streamlit run web/app_streamlit.py
```

What you get:

- Upload video or reuse `input/input.mp4`
- Select exercise type
- Run analysis from a web interface
- See output video preview + metrics + rep details

---

## Available Test Videos

| Video        | Duration | Expected Reps | Status             |
| ------------ | -------- | ------------- | ------------------ |
| `input.mp4`  | 40 sec   | 4             | ✅ Working         |
| `long.mp4`   | 60 sec   | ~30           | ✅ Detecting (12+) |
| `input2.mp4` | ?        | ?             | Available          |
| `2.mp4`      | ?        | ?             | Available          |
| `wrong.mp4`  | ?        | ?             | Available          |

---

## System Architecture

```
video_processor.py (Main)
├── PoseEstimator (MediaPipe BlazePose) → Detects 33 body landmarks
├── AngleEngine → Calculates joint angles (elbow, hip, knee, etc.)
├── TemporalEngine → Smooths angles (5-frame buffer + outlier rejection)
├── RepCounter → Detects pushup reps (UP/DOWN state machine)
└── FormAnalyzer → Analyzes form quality per rep
    └── form_standards.py → Injury-prevention angle rules
```

---

## Form Quality Feedback

### Joint Angle Standards (Correct Push-up Form)

| Joint              | Optimal | Good Range | Warning                                  |
| ------------------ | ------- | ---------- | ---------------------------------------- |
| **Elbow (bottom)** | 90°     | 75-95°     | <70° (hyperextension) or >110° (shallow) |
| **Hip**            | 175°    | 170-185°   | <150° (sagging) or >190° (pike)          |
| **Knee**           | 180°    | 175-180°   | <170° (bent legs)                        |

### Injury Indicators 🚨

- **Elbow Hyperextension**: Angle < 70° → Joint damage risk
- **Sagging Hips**: Hip angle < 150° → Spine stress
- **Pike Position**: Hip angle > 190° → Shoulder overload
- **Shallow Reps**: Insufficient depth → Not a valid rep

---

## Key Features

✅ **Accurate Rep Detection**

- Simple UP/DOWN state machine (reliable)
- Thresholds: DOWN at <120°, UP at >150°

✅ **Comprehensive Form Analysis**

- Quality score (0-100) per rep
- Elbow, hip, knee angle measurements
- Injury risk assessment
- Specific corrections for each issue

✅ **Adaptive to Different People**

- Works with any height/body proportions
- Auto-calibrates to individual's motion range

✅ **Real-time Feedback**

- Per-frame angle tracking
- Per-rep quality report
- Cumulative statistics

---

## Troubleshooting

**No reps detected?**

- Check that video shows someone doing pushups
- Ensure good lighting and clear body visibility
- Try a different input video

**Video processing is slow?**

- This is normal for full processing with MediaPipe
- Processing speed depends on frame rate and resolution

**Error: Module not found?**

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

---

## Files Overview

### Core (video_processor.py)

Main entry point that orchestrates the entire pipeline

### src/ (Python Modules)

- `pose_estimator.py` - MediaPipe BlazePose integration
- `angle_engine.py` - Vector angle calculations
- `temporal_engine.py` - Angle smoothing & outlier rejection
- `rep_counter.py` - Rep detection state machine
- `form_analyzer.py` - Form quality analysis per rep
- `form_standards.py` - Injury-prevention angle rules
- `rep_analyzer.py` - Rep metrics collection
- `exercise_rules.py` - Push-up validation rules

### Data

- `assests/reference_pushup.json` - Reference angle data
- Input videos: `input.mp4`, `long.mp4`, etc.

---

## Next Steps

1. **Run on input.mp4** → Verify basic setup works
2. **Run on long.mp4** → Test longer video
3. **Test on your own video** → Verify form coaching works for you
4. **Tune thresholds** (if needed) → Adjust in `rep_counter.py` if detection is off

---

**Last Updated:** March 25, 2026  
**Status:** ✅ Fully Functional - Rep Detection + Form Analysis Ready
its not checking the last few frames and how is the head position caliberated and what is the required head position and its not detecting if a person is doing a rep faster like around 30+ reps in 40sec
