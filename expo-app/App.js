import React, { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Linking,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { StatusBar } from "expo-status-bar";
import * as DocumentPicker from "expo-document-picker";
import * as ImagePicker from "expo-image-picker";
import Constants from "expo-constants";
import { Video, ResizeMode } from "expo-av";

import { API_BASE_URL, EXERCISES } from "./src/config";

const ISSUE_TO_TIP = {
  shallow_depth: "Increase range of motion and hit consistent depth each rep.",
  short_range_of_motion: "Complete full contraction and extension for each rep.",
  no_lockout: "Finish each rep fully at the top before the next rep.",
  hip_sag: "Brace the core and keep torso alignment stable.",
  forward_lean: "Keep torso more upright to reduce compensation.",
  inconsistent_depth: "Use a steady tempo and match depth across reps.",
  momentum_cheat: "Slow down and reduce momentum-driven movement.",
  neck_strain: "Keep neck neutral and avoid pulling with the neck.",
  knee_valgus: "Track knees in line with toes through movement.",
  core_instability: "Set your core before each rep and control transitions.",
  joint_stress: "Avoid forcing end range and keep movement controlled.",
};

function Metric({ label, value }) {
  return (
    <View style={styles.metricCard}>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={styles.metricValue}>{value}</Text>
    </View>
  );
}

function formatUnixTimestamp(value) {
  if (!value) return "-";
  const millis = Number(value) * 1000;
  if (!Number.isFinite(millis) || millis <= 0) return "-";
  return new Date(millis).toLocaleString();
}

function normalizeBaseUrl(value) {
  return String(value || "").trim().replace(/\/+$/, "");
}

function inferApiBaseUrl() {
  const candidates = [
    Constants?.expoConfig?.hostUri,
    Constants?.manifest?.debuggerHost,
    Constants?.manifest2?.extra?.expoClient?.hostUri,
  ];

  for (const candidate of candidates) {
    const text = String(candidate || "").trim();
    if (!text) continue;
    const host = text.split(":")[0];
    if (/^\d{1,3}(\.\d{1,3}){3}$/.test(host)) {
      return `http://${host}:8000`;
    }
  }
  return null;
}

async function fetchJsonWithTimeout(url, options = {}, timeoutMs = 180000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });

    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }

    if (!response.ok) {
      const detail = payload?.detail || `Server error ${response.status}`;
      throw new Error(detail);
    }

    return payload;
  } catch (err) {
    if (err?.name === "AbortError") {
      throw new Error("Request timed out. Check server status and try again.");
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

export default function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState(() => inferApiBaseUrl() || API_BASE_URL);
  const [availableExercises, setAvailableExercises] = useState(EXERCISES);
  const [exercise, setExercise] = useState("pushup");
  const [calibrationSeconds, setCalibrationSeconds] = useState("3");
  const [confidenceThreshold, setConfidenceThreshold] = useState("0.50");
  const [videoFile, setVideoFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [checkingServer, setCheckingServer] = useState(false);
  const [loadingRecent, setLoadingRecent] = useState(false);
  const [serverStatus, setServerStatus] = useState("unknown");
  const [serverMeta, setServerMeta] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [recentJobs, setRecentJobs] = useState([]);

  const summary = useMemo(() => result?.report?.summary || {}, [result]);
  const repReports = useMemo(() => result?.report?.rep_reports || [], [result]);

  const consistency = useMemo(() => {
    const total = Number(summary.total_reps || 0);
    const valid = Number(summary.valid_reps || 0);
    if (!total) return 0;
    return Math.round((valid / total) * 100);
  }, [summary]);

  const topIssues = useMemo(() => summary.top_issues || [], [summary]);
  const coachingTips = useMemo(() => {
    const tips = [];
    const existing = summary.coaching_tips || [];
    existing.forEach((tip) => {
      if (tip && !tips.includes(tip)) tips.push(String(tip));
    });

    const issueCounts = summary.issue_counts || {};
    Object.keys(issueCounts)
      .sort((a, b) => Number(issueCounts[b] || 0) - Number(issueCounts[a] || 0))
      .slice(0, 3)
      .forEach((key) => {
        const mapped = ISSUE_TO_TIP[key];
        if (mapped && !tips.includes(mapped)) tips.push(mapped);
      });

    if (!tips.length) {
      tips.push("No major issues detected. Keep your setup and pacing consistent.");
    }
    return tips.slice(0, 5);
  }, [summary]);

  const normalizedBaseUrl = useMemo(() => normalizeBaseUrl(apiBaseUrl), [apiBaseUrl]);

  const checkServerHealth = async () => {
    setCheckingServer(true);
    setError("");
    try {
      const payload = await fetchJsonWithTimeout(`${normalizedBaseUrl}/health`, {}, 12000);
      setServerStatus("online");
      setServerMeta(`max upload ${payload.max_upload_mb || "?"} MB`);
    } catch (err) {
      // Auto-fallback to inferred LAN host if user-entered URL is stale.
      const inferred = inferApiBaseUrl();
      if (inferred && inferred !== normalizedBaseUrl) {
        try {
          const payload = await fetchJsonWithTimeout(`${normalizeBaseUrl(inferred)}/health`, {}, 12000);
          setApiBaseUrl(inferred);
          setServerStatus("online");
          setServerMeta(`auto-switched to ${inferred} • max upload ${payload.max_upload_mb || "?"} MB`);
          return;
        } catch {
          // Fall through to offline status.
        }
      }

      setServerStatus("offline");
      const message = err instanceof Error ? err.message : "Health check failed";
      setServerMeta(message);
    } finally {
      setCheckingServer(false);
    }
  };

  const loadExercises = async () => {
    try {
      const payload = await fetchJsonWithTimeout(`${normalizedBaseUrl}/exercises`, {}, 15000);
      if (Array.isArray(payload.items) && payload.items.length) {
        setAvailableExercises(payload.items);
        if (!payload.items.includes(exercise)) {
          setExercise(payload.items[0]);
        }
      }
    } catch {
      // Keep fallback list when API metadata endpoint is unavailable.
    }
  };

  const loadRecentJobs = async () => {
    setLoadingRecent(true);
    try {
      const payload = await fetchJsonWithTimeout(`${normalizedBaseUrl}/recent-jobs?limit=8`, {}, 20000);
      if (Array.isArray(payload.items)) {
        setRecentJobs(payload.items);
      }
    } catch {
      setRecentJobs([]);
    } finally {
      setLoadingRecent(false);
    }
  };

  const openJobFromHistory = async (jobId) => {
    setLoading(true);
    setError("");
    try {
      const payload = await fetchJsonWithTimeout(`${normalizedBaseUrl}/jobs/${jobId}`, {}, 20000);
      setResult({
        ok: true,
        job_id: payload.job_id,
        exercise: payload.exercise,
        report: {
          exercise: payload.exercise,
          summary: payload.summary || {},
          rep_reports: [],
        },
        output_video_url: payload.output_video_url,
        report_url: payload.report_url,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Could not load selected session.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkServerHealth();
    loadExercises();
    loadRecentJobs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pickVideo = async () => {
    setError("");
    const selected = await DocumentPicker.getDocumentAsync({
      type: ["video/*"],
      copyToCacheDirectory: true,
      multiple: false,
    });

    if (selected.canceled || !selected.assets?.length) {
      return;
    }
    setVideoFile(selected.assets[0]);
    setResult(null);
  };

  const pickVideoFromGallery = async () => {
    setError("");
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setError("Gallery permission is required to pick a video.");
      return;
    }

    const selected = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["videos"],
      allowsEditing: false,
      quality: 0.8,
    });

    if (selected.canceled || !selected.assets?.length) {
      return;
    }

    const asset = selected.assets[0];
    setVideoFile({
      uri: asset.uri,
      name: asset.fileName || `gallery_${Date.now()}.mp4`,
      mimeType: asset.mimeType || "video/mp4",
    });
    setResult(null);
  };

  const recordVideoWithCamera = async () => {
    setError("");
    const camera = await ImagePicker.requestCameraPermissionsAsync();
    const mic = await ImagePicker.requestMicrophonePermissionsAsync();

    if (!camera.granted || !mic.granted) {
      setError("Camera and microphone permissions are required to record video.");
      return;
    }

    const recorded = await ImagePicker.launchCameraAsync({
      mediaTypes: ["videos"],
      allowsEditing: false,
      videoMaxDuration: 180,
      quality: 0.8,
    });

    if (recorded.canceled || !recorded.assets?.length) {
      return;
    }

    const asset = recorded.assets[0];
    setVideoFile({
      uri: asset.uri,
      name: asset.fileName || `recorded_${Date.now()}.mp4`,
      mimeType: asset.mimeType || "video/mp4",
    });
    setResult(null);
  };

  const runAnalysis = async () => {
    const parsedCalibration = Number.parseInt(calibrationSeconds, 10);
    if (Number.isNaN(parsedCalibration) || parsedCalibration < 1 || parsedCalibration > 15) {
      setError("Calibration must be between 1 and 15 seconds.");
      return;
    }

    const parsedConfidence = Number.parseFloat(confidenceThreshold);
    if (Number.isNaN(parsedConfidence) || parsedConfidence < 0.3 || parsedConfidence > 0.9) {
      setError("Confidence threshold must be between 0.30 and 0.90.");
      return;
    }

    if (!videoFile?.uri) {
      setError("Select a video before running analysis.");
      return;
    }

    if (typeof videoFile?.size === "number" && videoFile.size > 250 * 1024 * 1024) {
      setError("Selected video is over 250 MB. Choose a shorter/smaller clip.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const health = await fetchJsonWithTimeout(`${normalizedBaseUrl}/health`, {}, 10000);
      setServerStatus("online");
      setServerMeta(`max upload ${health?.max_upload_mb || "?"} MB`);

      const form = new FormData();
      form.append("exercise", exercise);
      form.append("calibration_seconds", String(parsedCalibration));
      form.append("confidence_threshold", String(parsedConfidence));
      form.append("file", {
        uri: videoFile.uri,
        name: videoFile.name || "input.mp4",
        type: videoFile.mimeType || "video/mp4",
      });

      const payload = await fetchJsonWithTimeout(`${normalizedBaseUrl}/analyze`, {
        method: "POST",
        body: form,
      }, 120000);
      setResult(payload);
      loadRecentJobs();
    } catch (err) {
      setServerStatus("offline");
      const message = err instanceof Error ? err.message : "Unexpected error while analyzing.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>FormAI Coach Mobile</Text>
        <Text style={styles.subtitle}>Static analysis mode for reliable expo demos</Text>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>Server Setup</Text>
          <TextInput
            value={apiBaseUrl}
            onChangeText={setApiBaseUrl}
            autoCapitalize="none"
            autoCorrect={false}
            placeholder="http://<your-laptop-ip>:8000"
            style={styles.input}
          />
          <View style={styles.rowGap}>
            <Pressable style={styles.secondaryButton} onPress={checkServerHealth}>
              {checkingServer ? (
                <ActivityIndicator color="#0b4f71" />
              ) : (
                <Text style={styles.secondaryButtonText}>Check API</Text>
              )}
            </Pressable>
            <Pressable style={styles.secondaryButton} onPress={loadExercises}>
              <Text style={styles.secondaryButtonText}>Refresh Exercises</Text>
            </Pressable>
          </View>
          <Text style={[styles.metaText, serverStatus === "online" ? styles.onlineText : styles.offlineText]}>
            Server: {serverStatus}
            {serverMeta ? ` • ${serverMeta}` : ""}
          </Text>
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>Recent Sessions</Text>
          <Pressable style={styles.secondaryButton} onPress={loadRecentJobs}>
            {loadingRecent ? <ActivityIndicator color="#0b4f71" /> : <Text style={styles.secondaryButtonText}>Refresh History</Text>}
          </Pressable>

          {recentJobs.length ? (
            recentJobs.map((job) => (
              <View key={job.job_id} style={styles.historyCard}>
                <Text style={styles.historyTitle}>Job {job.job_id}</Text>
                <Text style={styles.metaText}>Exercise: {job.exercise || "-"}</Text>
                <Text style={styles.metaText}>Created: {formatUnixTimestamp(job.created_at)}</Text>
                <Text style={styles.metaText}>
                  Reps: {job?.summary?.total_reps ?? 0} | Valid: {job?.summary?.valid_reps ?? 0} | Avg: {Math.round(Number(job?.summary?.avg_score || 0))}
                </Text>
                <View style={styles.linkRow}>
                  <Pressable onPress={() => openJobFromHistory(job.job_id)}>
                    <Text style={styles.linkText}>Load</Text>
                  </Pressable>
                  {job.output_video_url ? (
                    <Pressable onPress={() => Linking.openURL(job.output_video_url)}>
                      <Text style={styles.linkText}>Video</Text>
                    </Pressable>
                  ) : null}
                  {job.report_url ? (
                    <Pressable onPress={() => Linking.openURL(job.report_url)}>
                      <Text style={styles.linkText}>Report</Text>
                    </Pressable>
                  ) : null}
                </View>
              </View>
            ))
          ) : (
            <Text style={styles.metaText}>No recent sessions found yet.</Text>
          )}
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>1. Select Exercise</Text>
          <View style={styles.chipWrap}>
            {availableExercises.map((item) => {
              const active = item === exercise;
              return (
                <Pressable
                  key={item}
                  onPress={() => setExercise(item)}
                  style={[styles.chip, active && styles.chipActive]}
                >
                  <Text style={[styles.chipText, active && styles.chipTextActive]}>{item}</Text>
                </Pressable>
              );
            })}
          </View>

          <Text style={styles.sectionTitle}>2. Pick Video</Text>
          <View style={styles.rowGapWrap}>
            <Pressable style={styles.primaryButton} onPress={pickVideoFromGallery}>
              <Text style={styles.primaryButtonText}>Open Gallery</Text>
            </Pressable>
            <Pressable style={[styles.primaryButton, styles.cameraButton]} onPress={recordVideoWithCamera}>
              <Text style={styles.primaryButtonText}>Record Video</Text>
            </Pressable>
          </View>
          <Pressable style={[styles.secondaryWideButton]} onPress={pickVideo}>
            <Text style={styles.secondaryButtonText}>{videoFile ? "Browse Files Instead" : "Browse Files"}</Text>
          </Pressable>
          <Text style={styles.metaText}>
            {videoFile ? `Selected: ${videoFile.name || "video"}` : "No video selected yet"}
          </Text>

          <Text style={styles.sectionTitle}>3. Run Static Analysis</Text>
          <View style={styles.rowGap}>
            <View style={styles.flexOne}>
              <Text style={styles.inputLabel}>Calibration (1-15s)</Text>
              <TextInput
                value={calibrationSeconds}
                onChangeText={setCalibrationSeconds}
                keyboardType="numeric"
                style={styles.input}
              />
            </View>
            <View style={styles.flexOne}>
              <Text style={styles.inputLabel}>Confidence (0.30-0.90)</Text>
              <TextInput
                value={confidenceThreshold}
                onChangeText={setConfidenceThreshold}
                keyboardType="decimal-pad"
                style={styles.input}
              />
            </View>
          </View>
          <Pressable style={[styles.primaryButton, styles.runButton]} onPress={runAnalysis} disabled={loading}>
            {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.primaryButtonText}>Analyze</Text>}
          </Pressable>
          <Text style={styles.metaText}>API: {normalizedBaseUrl}</Text>
          {error ? <Text style={styles.errorText}>{error}</Text> : null}
        </View>

        {result ? (
          <View style={styles.panel}>
            <Text style={styles.sectionTitle}>Session Summary</Text>
            <View style={styles.metricGrid}>
              <Metric label="Total reps" value={String(summary.total_reps ?? 0)} />
              <Metric label="Valid reps" value={String(summary.valid_reps ?? 0)} />
              <Metric label="Avg score" value={String(Math.round(summary.avg_score ?? 0))} />
            </View>
            <Text style={styles.metaText}>Consistency: {consistency}%</Text>

            <Text style={styles.sectionTitle}>Top Issues</Text>
            {topIssues.length ? (
              <View style={styles.chipWrap}>
                {topIssues.map((issue) => (
                  <View key={issue} style={styles.issueChip}>
                    <Text style={styles.issueChipText}>{issue}</Text>
                  </View>
                ))}
              </View>
            ) : (
              <Text style={styles.metaText}>No major issues detected.</Text>
            )}

            <Text style={styles.sectionTitle}>Coaching Feedback</Text>
            <View style={styles.tipWrap}>
              {coachingTips.map((tip) => (
                <Text style={styles.tipText} key={tip}>• {tip}</Text>
              ))}
            </View>

            {result.output_video_url ? (
              <>
                <Text style={styles.sectionTitle}>Analyzed Output</Text>
                <Video
                  source={{ uri: result.output_video_url }}
                  style={styles.video}
                  useNativeControls
                  resizeMode={ResizeMode.CONTAIN}
                  isLooping={false}
                />
              </>
            ) : null}

            <View style={styles.linkRow}>
              {result.output_video_url ? (
                <Pressable onPress={() => Linking.openURL(result.output_video_url)}>
                  <Text style={styles.linkText}>Open output video</Text>
                </Pressable>
              ) : null}
              {result.report_url ? (
                <Pressable onPress={() => Linking.openURL(result.report_url)}>
                  <Text style={styles.linkText}>Open report JSON</Text>
                </Pressable>
              ) : null}
            </View>

            <Text style={styles.sectionTitle}>Rep Breakdown</Text>
            {repReports.length ? (
              repReports.slice(0, 8).map((rep) => {
                const classification = rep.classification || "Poor";
                const classificationColor = {
                  "Perfect": "#4ade80",
                  "Acceptable": "#60a5fa",
                  "Poor": "#fbbf24",
                  "Dangerous": "#ef4444",
                }[classification] || "#fbbf24";
                return (
                  <View key={`rep-${rep.rep_number}`} style={[styles.repCard, { borderLeftColor: classificationColor, borderLeftWidth: 4 }]}>
                    <Text style={styles.repTitle}>
                      Rep {rep.rep_number} • Score {Math.round(Number(rep.score || 0))}/10 • <Text style={{ color: classificationColor, fontWeight: "bold" }}>{classification}</Text>
                    </Text>
                    {(rep.feedback || []).slice(0, 2).map((line, idx) => (
                      <Text style={styles.repLine} key={`rep-${rep.rep_number}-${idx}`}>
                        - {line}
                      </Text>
                    ))}
                  </View>
                );
              })
            ) : (
              <Text style={styles.metaText}>No rep-level details found in response.</Text>
            )}
          </View>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: "#f5f8fa",
  },
  container: {
    padding: 16,
    paddingBottom: 28,
    gap: 12,
  },
  title: {
    fontSize: 28,
    fontWeight: "800",
    color: "#0d3c55",
  },
  subtitle: {
    marginTop: 4,
    fontSize: 14,
    color: "#4b6575",
  },
  panel: {
    backgroundColor: "#ffffff",
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: "#d8e6ef",
    gap: 10,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: "#143f56",
    marginTop: 4,
  },
  input: {
    borderWidth: 1,
    borderColor: "#c9dce8",
    borderRadius: 10,
    backgroundColor: "#f9fcfe",
    paddingHorizontal: 10,
    paddingVertical: 9,
    color: "#1e455b",
    fontSize: 14,
  },
  inputLabel: {
    fontSize: 12,
    color: "#4e6778",
    marginBottom: 4,
    fontWeight: "600",
  },
  rowGap: {
    flexDirection: "row",
    gap: 10,
  },
  rowGapWrap: {
    flexDirection: "row",
    gap: 10,
  },
  flexOne: {
    flex: 1,
  },
  chipWrap: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  chip: {
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "#b7cedc",
    paddingVertical: 7,
    paddingHorizontal: 12,
    backgroundColor: "#f6fbff",
  },
  chipActive: {
    backgroundColor: "#0d597e",
    borderColor: "#0d597e",
  },
  chipText: {
    fontSize: 12,
    color: "#274d63",
    fontWeight: "600",
  },
  chipTextActive: {
    color: "#ffffff",
  },
  primaryButton: {
    backgroundColor: "#0b4f71",
    borderRadius: 10,
    paddingVertical: 11,
    alignItems: "center",
    justifyContent: "center",
  },
  runButton: {
    backgroundColor: "#10755a",
  },
  cameraButton: {
    backgroundColor: "#8b4a00",
  },
  secondaryButton: {
    flex: 1,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#b7cedc",
    backgroundColor: "#f5faff",
    paddingVertical: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  secondaryButtonText: {
    color: "#1e526e",
    fontWeight: "700",
    fontSize: 13,
  },
  secondaryWideButton: {
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#b7cedc",
    backgroundColor: "#f5faff",
    paddingVertical: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  primaryButtonText: {
    color: "#ffffff",
    fontWeight: "700",
    fontSize: 15,
  },
  metaText: {
    fontSize: 12,
    color: "#4e6778",
  },
  errorText: {
    color: "#b42318",
    fontSize: 13,
    fontWeight: "600",
  },
  metricGrid: {
    flexDirection: "row",
    gap: 10,
  },
  issueChip: {
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "#efd7bd",
    backgroundColor: "#fff6eb",
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  issueChipText: {
    color: "#8a4b13",
    fontSize: 12,
    fontWeight: "700",
  },
  tipWrap: {
    gap: 6,
  },
  tipText: {
    color: "#21495f",
    fontSize: 13,
    lineHeight: 18,
  },
  metricCard: {
    flex: 1,
    backgroundColor: "#f6fbff",
    borderWidth: 1,
    borderColor: "#d9e8f2",
    borderRadius: 10,
    padding: 10,
  },
  metricLabel: {
    fontSize: 12,
    color: "#4b6575",
  },
  metricValue: {
    marginTop: 2,
    fontSize: 20,
    color: "#10374a",
    fontWeight: "800",
  },
  video: {
    width: "100%",
    height: 240,
    borderRadius: 12,
    backgroundColor: "#000",
  },
  linkRow: {
    marginTop: 8,
    flexDirection: "row",
    gap: 18,
    flexWrap: "wrap",
  },
  linkText: {
    color: "#0d5f96",
    fontWeight: "700",
  },
  repCard: {
    borderWidth: 1,
    borderColor: "#d9e8f2",
    borderRadius: 10,
    padding: 10,
    backgroundColor: "#f9fcff",
    marginBottom: 8,
  },
  repTitle: {
    color: "#1b475e",
    fontWeight: "700",
    fontSize: 13,
  },
  repLine: {
    marginTop: 4,
    color: "#35566b",
    fontSize: 12,
  },
  historyCard: {
    borderWidth: 1,
    borderColor: "#d9e8f2",
    borderRadius: 10,
    padding: 10,
    backgroundColor: "#f9fcff",
    marginTop: 8,
  },
  historyTitle: {
    color: "#1b475e",
    fontWeight: "700",
    fontSize: 13,
    marginBottom: 4,
  },
  onlineText: {
    color: "#0f8b3d",
  },
  offlineText: {
    color: "#b42318",
  },
});
