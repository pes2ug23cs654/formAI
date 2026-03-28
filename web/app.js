(function () {
  const $ = (id) => document.getElementById(id);

  const btnLive = $("btn-mode-live");
  const btnStatic = $("btn-mode-static");
  const staticControls = $("static-controls");
  const liveControls = $("live-controls");
  const viewLive = $("view-live");
  const viewStatic = $("view-static");
  const exerciseEl = $("exercise");
  const cam = $("cam");
  const cap = $("cap");
  const outLive = $("out-live");
  const outStatic = $("out-static");
  const btnStart = $("btn-start-live");
  const btnStop = $("btn-stop-live");
  const btnProcessStatic = $("btn-process-static");
  const staticStatus = $("static-status");
  const liveStatus = $("live-status");
  const mReps = $("m-reps");
  const mFeedback = $("m-feedback");
  const mScore = $("m-score");
  const mLive = $("m-live");
  const feedbackList = $("feedback-list");
  const stageHeading = $("stage-heading");
  const stageSub = $("stage-sub");
  const frameShell = document.querySelector("#view-live .frame-shell");

  let mode = "live";
  let sessionId = null;
  let stream = null;
  let rafId = null;
  let lastFrameTime = 0;
  const FRAME_INTERVAL_MS = 80;

  function setMode(next) {
    mode = next;
    const live = next === "live";
    btnLive.classList.toggle("pill--active", live);
    btnStatic.classList.toggle("pill--active", !live);
    staticControls.classList.toggle("hidden", live);
    liveControls.classList.toggle("hidden", !live);
    viewLive.classList.toggle("hidden", !live);
    viewStatic.classList.toggle("hidden", live);
    if (live) {
      stageHeading.textContent = "Live analysis";
      stageSub.textContent = "Annotated stream from your session — camera runs in the background only.";
    } else {
      stageHeading.textContent = "Annotated video";
      stageSub.textContent = "Upload a clip to process on the server; playback appears here when ready.";
    }
  }

  btnLive.addEventListener("click", () => setMode("live"));
  btnStatic.addEventListener("click", () => setMode("static"));

  function scoreFromLines(lines) {
    if (!lines || !lines.length) return null;
    for (const line of lines) {
      const m =
        line.match(/Score:\s*(\d+)\s*\/\s*100/i) ||
        line.match(/Quality:\s*(\d+)\s*\/\s*100/i);
      if (m) return `${m[1]}/100`;
    }
    return null;
  }

  function feedbackSummary(lines) {
    if (!lines || !lines.length) return "—";
    const skip = (s) =>
      /^\s*score:/i.test(s) || /^\s*quality:/i.test(s) || /^\s*rep\s*\d+/i.test(s);
    const body = lines.map((s) => s.replace(/^\[GOOD\]\s*/i, "").replace(/^\[ISSUE\]\s*/i, "").trim());
    const parts = [];
    for (const line of body) {
      if (!line || skip(line)) continue;
      parts.push(line);
      if (parts.join(" · ").length > 120) break;
    }
    if (!parts.length) return "—";
    const text = parts.join(" · ");
    return text.length > 140 ? `${text.slice(0, 137)}…` : text;
  }

  function renderCoach(data) {
    mReps.textContent = data.reps != null ? String(data.reps) : "—";
    const lines = data.feedback_lines || [];
    const sc = data.score != null ? String(data.score) : scoreFromLines(lines);
    mScore.textContent = sc || "—";
    mFeedback.textContent = data.feedback_summary != null ? data.feedback_summary : feedbackSummary(lines);
    mLive.textContent = data.live_msg || "";
    feedbackList.innerHTML = "";
    (data.feedback_lines || []).forEach((line) => {
      const li = document.createElement("li");
      const lower = line.toLowerCase();
      li.textContent = line;
      if (lower.includes("issue") || lower.includes("sag") || lower.includes("shallow")) {
        li.classList.add("issue");
      }
      feedbackList.appendChild(li);
    });
  }

  async function createSession() {
    const exercise = exerciseEl.value;
    const r = await fetch(`/api/session?exercise=${encodeURIComponent(exercise)}`, {
      method: "POST",
    });
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    sessionId = j.session_id;
  }

  async function deleteSession() {
    if (!sessionId) return;
    try {
      await fetch(`/api/session/${sessionId}`, { method: "DELETE" });
    } catch (_) {}
    sessionId = null;
  }

  async function sendFrame(blob) {
    if (!sessionId) return;
    const fd = new FormData();
    fd.append("session_id", sessionId);
    fd.append("file", blob, "frame.jpg");
    const r = await fetch("/api/frame", { method: "POST", body: fd });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(t || r.statusText);
    }
    const data = await r.json();
    outLive.src = `data:image/jpeg;base64,${data.image_b64}`;
    if (frameShell) frameShell.classList.add("has-frame");
    renderCoach(data);
  }

  function loop(ts) {
    rafId = requestAnimationFrame(loop);
    if (!stream || !sessionId) return;
    if (ts - lastFrameTime < FRAME_INTERVAL_MS) return;
    lastFrameTime = ts;
    const ctx = cap.getContext("2d");
    const w = cap.width;
    const h = cap.height;
    ctx.drawImage(cam, 0, 0, w, h);
    cap.toBlob(
      (blob) => {
        if (blob) sendFrame(blob).catch((e) => (liveStatus.textContent = String(e.message)));
      },
      "image/jpeg",
      0.85
    );
  }

  btnStart.addEventListener("click", async () => {
    liveStatus.textContent = "";
    btnStart.disabled = true;
    try {
      await deleteSession();
      await createSession();
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      cam.srcObject = stream;
      await cam.play();
      const track = stream.getVideoTracks()[0];
      const settings = track.getSettings();
      if (settings.width && settings.height) {
        cap.width = settings.width;
        cap.height = settings.height;
      }
      liveStatus.textContent = "Streaming… allow camera if prompted.";
      btnStop.disabled = false;
      rafId = requestAnimationFrame(loop);
    } catch (e) {
      liveStatus.textContent = e.message || String(e);
      btnStart.disabled = false;
      await deleteSession();
    }
  });

  btnStop.addEventListener("click", async () => {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
      cam.srcObject = null;
    }
    await deleteSession();
    btnStart.disabled = false;
    btnStop.disabled = true;
    liveStatus.textContent = "Stopped.";
    outLive.removeAttribute("src");
    if (frameShell) frameShell.classList.remove("has-frame");
  });

  btnProcessStatic.addEventListener("click", async () => {
    const f = $("video-file").files[0];
    if (!f) {
      staticStatus.textContent = "Choose a video file first.";
      return;
    }
    staticStatus.textContent = "Uploading…";
    outStatic.removeAttribute("src");
    const fd = new FormData();
    fd.append("file", f);
    fd.append("exercise", exerciseEl.value);
    const up = await fetch("/api/process-static", { method: "POST", body: fd });
    if (!up.ok) {
      staticStatus.textContent = await up.text();
      return;
    }
    const { job_id: jobId } = await up.json();
    staticStatus.textContent = "Processing… this can take a while.";
    const poll = setInterval(async () => {
      const st = await fetch(`/api/job/${jobId}`);
      if (!st.ok) {
        clearInterval(poll);
        staticStatus.textContent = "Status check failed.";
        return;
      }
      const j = await st.json();
      if (j.status === "done") {
        clearInterval(poll);
        staticStatus.textContent = "Done. Play below.";
        if (j.summary) {
          renderCoach({
            reps: j.summary.reps,
            feedback_lines: j.summary.feedback_lines || [],
            live_msg: "",
          });
        }
        outStatic.onerror = () => {
          staticStatus.textContent =
            "Video failed to load in the browser. Re-run after: pip install imageio-ffmpeg (or install ffmpeg), then process again.";
        };
        outStatic.onloadeddata = () => {
          staticStatus.textContent = "Done. Play below.";
        };
        outStatic.src = `/api/output/${jobId}?t=${Date.now()}`;
        outStatic.load();
      } else if (j.status === "error") {
        clearInterval(poll);
        staticStatus.textContent = j.error || "Processing failed.";
      }
    }, 1000);
  });

  renderCoach({ reps: 0, feedback_lines: [], live_msg: "" });
})();
