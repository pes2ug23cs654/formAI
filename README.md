# FormAI

**FormAI** is a computer-vision coaching assistant that estimates pose from video, counts repetitions, scores movement quality, and returns actionable feedback for **push-ups** and **squats**. It ships with a **command-line pipeline** for batch video processing and a **browser-based UI** for live webcam sessions and uploaded clips.

---

## Table of contents

- [Why FormAI](#why-formai)
- [Features](#features)
- [Tech stack](#tech-stack)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI — video file](#cli--video-file)
  - [CLI — webcam](#cli--webcam)
  - [Web application](#web-application)
- [Web UI overview](#web-ui-overview)
- [API reference (summary)](#api-reference-summary)
- [Architecture](#architecture)
- [Scoring and biomechanics](#scoring-and-biomechanics)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

---

## Why FormAI

Most fitness tools stop at *which* exercise you are doing. FormAI focuses on *how* you move: joint angles, depth, alignment, and repeatability—so feedback targets injury risk and better reps, not just rep counts.

---

## Features

| Area | Details |
|------|---------|
| **Exercises** | Push-up and squat pipelines with exercise-specific rep logic and analyzers. |
| **Pose** | [MediaPipe](https://developers.google.com/mediapipe) BlazePose—33 landmarks with visibility and depth (`z`) for stable angles. |
| **Per-rep feedback** | Score (0–100), positive cues, and issues after each counted rep. |
| **Session intelligence** | Push-ups: aggregated session report (averages, recurring issues, trends, coach tips via `SessionAnalyzer`). Squats: per-rep analysis and issue rollups. |
| **Visualization** | Skeleton overlay, rep count, stage labels, and live squat coaching cues on the rendered frames. |
| **CLI** | Process any supported video to an annotated file; optional live OpenCV window from the default camera. |
| **Web UI** | **Live**: browser webcam (camera capture is off-screen; user sees **annotated frames only**). **Upload**: server-side processing, **browser-playable MP4** (H.264 re-encode via FFmpeg / `imageio-ffmpeg`). |
| **Coach panel** | Reps, score, feedback summary, live cue banner, and expandable detail lines (live: streamed; upload: post-job summary). |

---

## Tech stack

- **Python 3.11+** (recommended)
- **OpenCV** — video I/O, drawing, encoding
- **MediaPipe** — pose estimation
- **NumPy** — geometry and smoothing
- **FastAPI + Uvicorn** — web server and API
- **python-multipart** — uploads
- **imageio-ffmpeg** — bundled FFmpeg binary for **H.264** exports (HTML5-compatible MP4 in the browser)

---

## Requirements

- Python **3.11+**
- Webcam with decent lighting (live / CLI webcam modes)
- For **web upload playback**: `imageio-ffmpeg` (installed via `requirements.txt`) or a system **ffmpeg** on `PATH` for re-encoding

---

## Installation

```bash
git clone <your-repo-url>
cd formAI

python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### CLI — video file

```bash
python app.py <input_video> <output_video> [exercise]
```

- **`exercise`**: `pushup` (default) or `squat`
- Output video uses OpenCV’s internal MPEG-4 writer (`mp4v`) unless you use the web pipeline (see below).

**Examples**

```bash
python app.py input.mp4 output.mp4
python app.py squats.mp4 out_squat.mp4 squat
```

### CLI — webcam

```bash
python app.py --webcam [exercise]
```

**Examples**

```bash
python app.py --webcam
python app.py --webcam squat
```

Press **`q`** in the OpenCV window to quit.

### Web application

From the project root (with `venv` activated):

```bash
python -m uvicorn ui_app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in a modern browser (Chrome, Edge, or Firefox recommended).

---

## Web UI overview

- **Mode — Live**  
  Start the session to grant camera access. Frames are sent to the server for pose and coaching; the page shows **only** the **annotated** stream plus the **Coach** sidebar (reps, score, feedback, cues, details).

- **Mode — Upload video**  
  Choose a file (e.g. `.mp4`, `.mov`, `.avi`, `.webm`). The server runs the same analysis pipeline as the CLI, re-encodes the result to **H.264** for in-browser playback, then shows **session-level** reps, score, and feedback when the job completes.

- **Movement**  
  Select **Push-up** or **Squat** before starting live or processing a file.

> **Note:** In upload mode, coach metrics populate when processing **finishes**, not while scrubbing the player. Live mode updates after each detected rep (or continuously for alignment cues in squats).

---

## API reference (summary)

Base URL: same host as the UI (e.g. `http://127.0.0.1:8000`).

| Method | Path | Purpose |
|--------|------|--------|
| `GET` | `/` | Serves the web UI |
| `POST` | `/api/session?exercise=pushup` | Create a live session; returns `session_id` |
| `DELETE` | `/api/session/{session_id}` | End session |
| `POST` | `/api/frame` | `multipart/form-data`: `session_id`, `file` (JPEG)—returns JSON + base64 annotated frame |
| `POST` | `/api/process-static` | `file` + `form` field `exercise`—returns `job_id` |
| `GET` | `/api/job/{job_id}` | `status`, `error`, and when `done`, `summary` (`reps`, `feedback_lines`) |
| `GET` | `/api/output/{job_id}` | Annotated MP4 when job is `done` |

Static files are under `/static/`.

---

## Architecture

```
app.py                          # CLI entry (process_video / process_webcam)
ui_app.py                       # FastAPI app, sessions, jobs, static UI
web/                            # index.html, styles.css, app.js
video_processor.py              # Batch pipeline, LivePipeline (live/UI frames),
                                # FFmpeg re-encode for web-safe MP4 when web_playback=True
src/
  pose_estimator.py             # MediaPipe capture + skeleton draw
  angle_engine.py               # Joint angles (elbow, hip, knee, …)
  temporal_engine.py            # Smoothing / stages (push-up)
  squat_temporal_engine.py      # Squat-specific temporal logic
  rep_counter.py                # Push-up rep state machine
  squat_rep_counter.py          # Squat rep state machine
  form_analyzer.py              # Push-up per-rep scoring and text feedback
  squat_analyzer.py             # Squat per-rep scoring and issues
  form_standards.py             # Biomechanical thresholds
  session_analyzer.py           # Multi-rep session rollup (push-ups)
```

**Batch flow:** read frame → pose → angles → temporal smooth → rep detection → on rep complete, analyze → draw overlay → write frame.

**Live flow:** same logic via `LivePipeline.process(frame)`; the web client posts JPEGs and displays the returned annotated image.

---

## Scoring and biomechanics

Rules live primarily in `src/form_standards.py` and the analyzers. High-level expectations:

| Signal | Role |
|--------|------|
| **Elbow (push-up bottom)** | Typical good band ~75–95°; shallow or hyperextended depth flagged. |
| **Hip alignment** | Neutral plank ~170–185°; sag or pike flagged. |
| **Head (3D ratio)** | Head–shoulder vs shoulder–hip distance ratio to reduce camera-distance bias. |
| **Knees (push-up)** | Legs reasonably straight in standard push-up. |
| **Squats** | Knee angle path, depth, torso, hip hinge; issues summarized per rep and in upload summaries. |

Exact strings and score deductions are implemented in code; tune thresholds in `form_standards.py`, `rep_counter.py`, and squat modules if your use case requires it.

### Example CLI console output (push-up)

Per-rep blocks print quality lines and `[ISSUE]` / good markers. After the video, a **session summary** can include overall score, primary issues by frequency, trend, and bullet coach tips.

---

## Project layout

```
formAI/
├── app.py
├── ui_app.py
├── video_processor.py
├── requirements.txt
├── README.md
├── web/                 # Front-end assets
├── web_data/            # Created at runtime (uploads, outputs)—gitignored if configured
├── src/                 # Library modules
└── venv/                # Local virtualenv (not committed)
```

Sample media in the repo (e.g. `input.mp4`, squat clips) are useful for smoke tests; add your own videos for production trials.

---

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| **Pose not detected** | Improve lighting; full body in frame; unobstructed limbs. |
| **Reps not counted** | Ensure side or diagonal view for elbows/knees; movement should cross configured angle thresholds; try a longer clip. |
| **Upload plays black / 0:00 in browser** | Ensure `imageio-ffmpeg` is installed (`pip install -r requirements.txt`). Re-process the file so output is H.264 re-encoded. |
| **`ModuleNotFoundError`** | Activate `venv` and run `pip install -r requirements.txt`. |
| **Webcam denied** | Browser permissions; use HTTPS or `localhost` where required by the browser. |
| **CLI `app.py` errors on first argument** | First argument must be `--webcam` or an **input file path**; see [Usage](#usage). |

---

## Contributing

1. Fork the repository and create a feature branch.
2. Keep changes focused; match existing style and typing patterns.
3. Verify CLI paths (`pushup` / `squat`) and, if you touch the UI, smoke-test live and upload flows.
4. Open a pull request with a clear description of behavior changes.

Add an explicit **LICENSE** file at the repo root if you distribute the project publicly.

---

## Acknowledgments

- [Google MediaPipe](https://developers.google.com/mediapipe) for pose estimation  
- [OpenCV](https://opencv.org/) for video and visualization  
- [FastAPI](https://fastapi.tiangolo.com/) for the web layer  

---

<p>
  <sub>Built for reliable offline-friendly coaching: run locally, own your data, and iterate on rules in code.</sub>
</p>
