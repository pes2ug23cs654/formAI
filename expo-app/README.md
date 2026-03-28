FormAI Expo Mobile App

Overview

- This Expo app runs static workout video analysis by uploading a video to the Python backend API.
- The backend uses the same process_video pipeline from the main project.

What is implemented

- Video picker and upload flow
- Dynamic exercise list from backend
- API health check in-app
- Analysis settings (calibration and confidence)
- Session summary metrics
- Top issues and coaching tips
- Rep-level breakdown
- Processed output video playback
- Links to processed video and report JSON
- Recent sessions history with quick reload

Backend requirements

- Run from the project root folder (formAI)
- Python venv active with dependencies installed from requirements.txt

Backend start command

- c:/Users/HP/OneDrive/Desktop/tevin/sem6/hackathon/praxis/formAI/venv311/Scripts/python.exe -m uvicorn mobile_api:app --host 0.0.0.0 --port 8000

Expo start command

- In expo-app folder:
- npm install
- npx expo start --lan --port 8083

iPhone demo setup

1. Keep laptop and iPhone on the same Wi-Fi.
2. Confirm API URL in app is your laptop IP with port 8000.
3. Open Expo Go on iPhone and scan QR from Metro terminal.
4. Use Select Video and Analyze.

Notes

- Default API URL is set in src/config.js.
- If your IP changes, update it in the app Server Setup field.
- API endpoints used by app: /health, /exercises, /analyze, /recent-jobs, /jobs/{job_id}
