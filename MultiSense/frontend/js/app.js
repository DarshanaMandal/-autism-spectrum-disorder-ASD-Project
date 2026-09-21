/**
 * MultiSense Frontend Application Logic
 * Integrates FastAPI endpoints, video uploading, sensor telemetry,
 * preset simulations, and diagnostic report generation.
 */

// Auto-detect API Base URL or read custom endpoint from storage
let API_BASE = localStorage.getItem("multisense_api_url") || window.location.origin;
if (API_BASE.includes("localhost") || API_BASE.includes("127.0.0.1") || window.location.port === "8000") {
    API_BASE = window.location.origin;
} else if (!API_BASE.startsWith("http")) {
    API_BASE = "http://127.0.0.1:8000";
}

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initSettingsModal();
    initSensors();
    initKinematicSliders();
    initVideoUpload();
    initHardwareSensorTab();
    checkBackendHealth();
    checkMongoStatus();
    refreshSensorHistory();
    setInterval(() => {
        checkBackendHealth();
        checkMongoStatus();
    }, 8000);

    // Initialize visualizer pose animator on tab switch or immediately
    if (window.visualizer) {
        window.visualizer.initPoseAnimator();
    }
});

/* =========================================================
   1. BACKEND HEALTH & CONNECTIVITY MONITOR
   ========================================================= */
async function checkBackendHealth() {
    const statusPill = document.getElementById("backendStatusPill");
    const startTime = performance.now();

    try {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) throw new Error("Status " + res.status);
        const data = await res.json();
        const latency = Math.round(performance.now() - startTime);

        if (statusPill) {
            statusPill.innerHTML = `<span class="pulsing-dot"></span> Online (${latency}ms)`;
            statusPill.style.color = "var(--status-green)";
            statusPill.style.borderColor = "rgba(16, 185, 129, 0.3)";
        }
    } catch (err) {
        if (statusPill) {
            statusPill.innerHTML = `<span class="pulsing-dot" style="background: var(--status-red); box-shadow: none;"></span> Disconnected`;
            statusPill.style.color = "var(--status-red)";
            statusPill.style.borderColor = "rgba(239, 68, 68, 0.3)";
        }
    }
}

/* =========================================================
   2. TAB NAVIGATION
   ========================================================= */
function initTabs() {
    const buttons = document.querySelectorAll(".tab-btn");
    buttons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");

            // Update active states
            buttons.forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const targetContent = document.getElementById(targetId);
            if (targetContent) targetContent.classList.add("active");

            // Resize canvas if switching to tabs with canvas
            if (targetId === "tab-telemetry" && window.visualizer) {
                setTimeout(() => window.visualizer.resizeECG(), 50);
            } else if (targetId === "tab-gan" && window.visualizer) {
                setTimeout(() => window.visualizer.resizePose(), 50);
            }
        });
    });
}

/* =========================================================
   3. VIDEO UPLOAD & DEMO CLIPS INFERENCE
   ========================================================= */
function initVideoUpload() {
    const dropZone = document.getElementById("videoDropZone");
    const fileInput = document.getElementById("videoFileInput");

    if (dropZone && fileInput) {
        dropZone.addEventListener("click", () => fileInput.click());

        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        });

        dropZone.addEventListener("dragleave", () => {
            dropZone.classList.remove("dragover");
        });

        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                processSelectedVideo(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                processSelectedVideo(e.target.files[0]);
            }
        });
    }
}

async function loadDemoVideo(sampleName) {
    const videoUrl = `/samples/${sampleName}`;
    const previewWrapper = document.getElementById("videoPreviewWrapper");
    const videoPlayer = document.getElementById("videoPlayer");
    const btn = document.getElementById("analyzeVideoBtn");

    if (previewWrapper && videoPlayer) {
        previewWrapper.style.display = "block";
        videoPlayer.src = videoUrl;
        videoPlayer.play();
    }

    if (btn) {
        btn.innerHTML = `<span class="spinner"></span> Fetching Demo Video & Extracting Kinematics...`;
        btn.disabled = true;
    }

    try {
        // Fetch file blob from local server
        const res = await fetch(videoUrl);
        const blob = await res.blob();
        const file = new File([blob], sampleName, { type: "video/mp4" });
        await executeVideoClassification(file);
    } catch (err) {
        alert("Demo clip loading error: " + err.message);
    } finally {
        if (btn) {
            btn.innerHTML = `📹 Analyze Video Kinematics`;
            btn.disabled = false;
        }
    }
}

function processSelectedVideo(file) {
    const previewWrapper = document.getElementById("videoPreviewWrapper");
    const videoPlayer = document.getElementById("videoPlayer");
    const fileNameDisplay = document.getElementById("selectedFileName");
    const analyzeBtn = document.getElementById("analyzeVideoBtn");

    if (previewWrapper && videoPlayer) {
        previewWrapper.style.display = "block";
        videoPlayer.src = URL.createObjectURL(file);
        videoPlayer.play();
    }

    if (fileNameDisplay) {
        fileNameDisplay.innerText = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        fileNameDisplay.style.display = "block";
    }

    if (analyzeBtn) {
        analyzeBtn.onclick = () => executeVideoClassification(file);
    }
}

async function executeVideoClassification(file) {
    const analyzeBtn = document.getElementById("analyzeVideoBtn");
    const modelChoice = document.getElementById("videoModelChoice")?.value || "gan_augmented";

    if (analyzeBtn) {
        analyzeBtn.innerHTML = `⚙️ Processing Frames with MediaPipe & Random Forest...`;
        analyzeBtn.disabled = true;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`${API_BASE}/predict/video?model_choice=${modelChoice}`, {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || "Video analysis failed");
        }

        const data = await res.json();
        renderDiagnosis(data);

        // Update pose animator if possible
        if (window.visualizer && data.predicted_class) {
            window.visualizer.loadSampleSequence(data.predicted_class);
        }
    } catch (err) {
        alert("Classification Failed: " + err.message);
    } finally {
        if (analyzeBtn) {
            analyzeBtn.innerHTML = `📹 Analyze Video Kinematics`;
            analyzeBtn.disabled = false;
        }
    }
}

/* =========================================================
   4. MULTIMODAL TELEMETRY SENSOR SUITE
   ========================================================= */
function initSensors() {
    const hrSlider = document.getElementById("hrSlider");
    const spo2Slider = document.getElementById("spo2Slider");
    const accSlider = document.getElementById("accSlider");
    const gyroSlider = document.getElementById("gyroSlider");
    const luxSlider = document.getElementById("luxSlider");

    if (hrSlider) {
        hrSlider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            document.getElementById("hrDisplay").innerText = `${val.toFixed(0)} BPM`;
            if (window.visualizer) window.visualizer.setBPM(val);

            // Warning color
            const disp = document.getElementById("hrDisplay");
            disp.style.color = val >= 135 ? "var(--status-red)" : (val >= 115 ? "var(--status-amber)" : "var(--status-green)");
        });
    }

    if (spo2Slider) {
        spo2Slider.addEventListener("input", (e) => {
            document.getElementById("spo2Display").innerText = `${parseFloat(e.target.value).toFixed(1)}%`;
        });
    }

    if (accSlider) {
        accSlider.addEventListener("input", (e) => {
            document.getElementById("accDisplay").innerText = `${parseFloat(e.target.value).toFixed(1)} g`;
        });
    }

    if (gyroSlider) {
        gyroSlider.addEventListener("input", (e) => {
            document.getElementById("gyroDisplay").innerText = `${parseFloat(e.target.value).toFixed(2)} rad/s`;
        });
    }

    if (luxSlider) {
        luxSlider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            document.getElementById("luxDisplay").innerText = `${val.toFixed(0)} Lux`;
            const badge = document.getElementById("luxWarningBadge");
            if (badge) {
                badge.style.display = val >= 550 ? "inline-block" : "none";
            }
        });
    }
}

function applySensorPreset(scenario) {
    const hr = document.getElementById("hrSlider");
    const spo2 = document.getElementById("spo2Slider");
    const acc = document.getElementById("accSlider");
    const gyro = document.getElementById("gyroSlider");
    const lux = document.getElementById("luxSlider");

    if (scenario === "calm") {
        if (hr) hr.value = 76;
        if (spo2) spo2.value = 98.5;
        if (acc) acc.value = 1.1;
        if (gyro) gyro.value = 0.15;
        if (lux) lux.value = 320;
    } else if (scenario === "meltdown") {
        if (hr) hr.value = 146;
        if (spo2) spo2.value = 95.5;
        if (acc) acc.value = 4.9;
        if (gyro) gyro.value = 0.45;
        if (lux) lux.value = 680;
    } else if (scenario === "flapping") {
        if (hr) hr.value = 122;
        if (spo2) spo2.value = 97.4;
        if (acc) acc.value = 3.6;
        if (gyro) gyro.value = 0.52;
        if (lux) lux.value = 450;
    } else if (scenario === "spinning") {
        if (hr) hr.value = 108;
        if (spo2) spo2.value = 98.0;
        if (acc) acc.value = 2.3;
        if (gyro) gyro.value = 1.45;
        if (lux) lux.value = 380;
    }

    // Trigger input events
    [hr, spo2, acc, gyro, lux].forEach(s => s && s.dispatchEvent(new Event("input")));
}

async function executeMultimodalPrediction() {
    const btn = document.getElementById("predictMultimodalBtn");
    if (btn) {
        btn.innerHTML = `<span class="spinner"></span> Fusing Sensor Streams (Accuracy: 97.44%)...`;
        btn.disabled = true;
    }

    const payload = {
        wrist_velocity_mean: parseFloat(document.getElementById("accSlider")?.value) * 0.15 || 0.35,
        wrist_velocity_max: 0.95,
        wrist_velocity_std: 0.18,
        wrist_acceleration_mean: 0.22,
        arm_velocity_mean: 0.30,
        head_velocity_mean: (parseFloat(document.getElementById("hrSlider")?.value) > 130) ? 0.54 : 0.16,
        head_velocity_max: 0.35,
        head_pitch_variance: (parseFloat(document.getElementById("hrSlider")?.value) > 130) ? 0.38 : 0.04,
        shoulder_velocity_mean: 0.14,
        body_velocity_mean: 0.15,
        angular_velocity_centroid: parseFloat(document.getElementById("gyroSlider")?.value) * 0.4 || 0.12,
        movement_intensity: 0.45,
        pose_variance: 0.035,
        elbow_angle_mean: 95.0,
        elbow_angle_std: 15.0,
        shoulder_angle_mean: 65.0,
        shoulder_angle_std: 12.0,
        max30102_hr_bpm: parseFloat(document.getElementById("hrSlider")?.value) || 120.0,
        max30102_spo2_pct: parseFloat(document.getElementById("spo2Slider")?.value) || 97.5,
        mpu6050_acc_mag_g: parseFloat(document.getElementById("accSlider")?.value) || 3.2,
        mpu6050_gyro_yaw_rads: parseFloat(document.getElementById("gyroSlider")?.value) || 0.5,
        bh1750_ambient_lux: parseFloat(document.getElementById("luxSlider")?.value) || 450.0
    };

    try {
        const res = await fetch(`${API_BASE}/predict/multimodal`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Server returned status " + res.status);
        const data = await res.json();
        renderDiagnosis(data);

        if (window.visualizer && data.predicted_class) {
            window.visualizer.loadSampleSequence(data.predicted_class);
        }
    } catch (err) {
        alert("Multimodal Prediction Error: " + err.message);
    } finally {
        if (btn) {
            btn.innerHTML = `⚡ Execute Sensor Fusion Diagnosis`;
            btn.disabled = false;
        }
    }
}

/* =========================================================
   5. KINEMATIC FEATURE LAB
   ========================================================= */
function initKinematicSliders() {
    const sliders = [
        { id: "kinWristVel", disp: "kinWristVelDisp", unit: " m/s" },
        { id: "kinHeadPitch", disp: "kinHeadPitchDisp", unit: " rad²" },
        { id: "kinHeadVel", disp: "kinHeadVelDisp", unit: " m/s" },
        { id: "kinAngularVel", disp: "kinAngularVelDisp", unit: " rad/s" },
        { id: "kinIntensity", disp: "kinIntensityDisp", unit: "" }
    ];

    sliders.forEach(s => {
        const el = document.getElementById(s.id);
        if (el) {
            el.addEventListener("input", (e) => {
                const d = document.getElementById(s.disp);
                if (d) d.innerText = `${parseFloat(e.target.value).toFixed(2)}${s.unit}`;
                if (document.getElementById("livePredictToggle")?.checked) {
                    executeKinematicPrediction();
                }
            });
        }
    });
}

function applyKinematicPreset(preset) {
    const w = document.getElementById("kinWristVel");
    const hp = document.getElementById("kinHeadPitch");
    const hv = document.getElementById("kinHeadVel");
    const ang = document.getElementById("kinAngularVel");
    const inten = document.getElementById("kinIntensity");

    if (preset === "ArmFlapping") {
        if (w) w.value = 0.52;
        if (hp) hp.value = 0.03;
        if (hv) hv.value = 0.12;
        if (ang) ang.value = 0.08;
        if (inten) inten.value = 0.45;
    } else if (preset === "HeadBanging") {
        if (w) w.value = 0.16;
        if (hp) hp.value = 0.42;
        if (hv) hv.value = 0.58;
        if (ang) ang.value = 0.10;
        if (inten) inten.value = 0.68;
    } else if (preset === "Spinning") {
        if (w) w.value = 0.22;
        if (hp) hp.value = 0.04;
        if (hv) hv.value = 0.24;
        if (ang) ang.value = 0.75;
        if (inten) inten.value = 0.52;
    }

    // Trigger inputs
    [w, hp, hv, ang, inten].forEach(el => el && el.dispatchEvent(new Event("input")));
    executeKinematicPrediction();
}

async function executeKinematicPrediction() {
    const modelChoice = document.getElementById("kinModelChoice")?.value || "gan_augmented";
    const payload = {
        wrist_velocity_mean: parseFloat(document.getElementById("kinWristVel")?.value) || 0.45,
        wrist_velocity_max: (parseFloat(document.getElementById("kinWristVel")?.value) || 0.45) * 2.4,
        wrist_velocity_std: 0.18,
        wrist_acceleration_mean: 0.24,
        arm_velocity_mean: 0.38,
        head_velocity_mean: parseFloat(document.getElementById("kinHeadVel")?.value) || 0.15,
        head_velocity_max: 0.32,
        head_pitch_variance: parseFloat(document.getElementById("kinHeadPitch")?.value) || 0.04,
        shoulder_velocity_mean: 0.14,
        body_velocity_mean: 0.16,
        angular_velocity_centroid: parseFloat(document.getElementById("kinAngularVel")?.value) || 0.08,
        movement_intensity: parseFloat(document.getElementById("kinIntensity")?.value) || 0.42,
        pose_variance: 0.035,
        elbow_angle_mean: 95.0,
        elbow_angle_std: 15.0,
        shoulder_angle_mean: 65.0,
        shoulder_angle_std: 12.0,
        model_choice: modelChoice
    };

    try {
        const res = await fetch(`${API_BASE}/predict/features`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        renderDiagnosis(data);

        if (window.visualizer && data.predicted_class) {
            window.visualizer.loadSampleSequence(data.predicted_class);
        }
    } catch (err) {
        console.error("Kinematic prediction error:", err);
    }
}

/* =========================================================
   6. CONDITIONAL BI-GRU TIMEGAN GENERATOR
   ========================================================= */
async function executeGanGeneration() {
    const btn = document.getElementById("generateGanBtn");
    const behaviorClass = document.getElementById("ganClassSelect")?.value || "ArmFlapping";
    const sampleCount = parseInt(document.getElementById("ganCountInput")?.value || "5", 10);

    if (btn) {
        btn.innerHTML = `<span class="spinner"></span> Synthesizing ${sampleCount} sequences via Bi-GRU...`;
        btn.disabled = true;
    }

    try {
        const res = await fetch(`${API_BASE}/generate/synthetic`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                behavior_class: behaviorClass,
                num_samples: sampleCount
            })
        });

        if (!res.ok) throw new Error("GAN Generation Failed: " + res.status);
        const data = await res.json();

        // Update animation
        if (window.visualizer) {
            window.visualizer.loadSampleSequence(behaviorClass);
        }

        // Display summary
        const infoBox = document.getElementById("ganResultInfo");
        if (infoBox) {
            infoBox.innerHTML = `
                <strong>[✓] Generated ${data.num_samples} sequences</strong> (Shape: <code>${JSON.stringify(data.sequence_shape)}</code>)<br>
                Kinematics: Wrist Vel: <code>${data.kinematic_stats.wrist_velocity_mean.toFixed(2)} m/s</code>, 
                Head Pitch: <code>${data.kinematic_stats.head_pitch_variance.toFixed(3)}</code>,
                Intensity: <code>${data.kinematic_stats.movement_intensity.toFixed(2)}</code>
            `;
            infoBox.style.display = "block";
        }
    } catch (err) {
        alert("GAN Error: " + err.message);
    } finally {
        if (btn) {
            btn.innerHTML = `🧬 Synthesize Biomechanical Trajectories`;
            btn.disabled = false;
        }
    }
}

/* =========================================================
   7. DIAGNOSTIC HUB & CLINICAL REPORT RENDERER
   ========================================================= */
function renderDiagnosis(data) {
    const cls = data.predicted_class || "Unknown";
    const conf = (data.confidence * 100).toFixed(1);
    const probs = data.probabilities || {};

    // 1. Classification Badge
    const badge = document.getElementById("diagnosticBadge");
    if (badge) {
        badge.innerText = cls;
        badge.className = "classification-badge";
        if (cls === "ArmFlapping") badge.classList.add("badge-arm");
        else if (cls === "HeadBanging") badge.classList.add("badge-head");
        else if (cls === "Spinning") badge.classList.add("badge-spin");
        else badge.classList.add("badge-calm");
    }

    // 2. Confidence Readout
    const confEl = document.getElementById("confidenceReadout");
    if (confEl) confEl.innerText = `${conf}%`;

    const modelTag = document.getElementById("activeModelTag");
    if (modelTag) modelTag.innerText = data.model_used || "GAN-Augmented RF";

    // 3. Probability Bars
    const pArm = (probs["ArmFlapping"] || 0) * 100;
    const pHead = (probs["HeadBanging"] || 0) * 100;
    const pSpin = (probs["Spinning"] || 0) * 100;

    setBar("Arm", pArm);
    setBar("Head", pHead);
    setBar("Spin", pSpin);

    // 4. Clinical Alert Box
    const alertData = data.clinical_alert;
    const alertBox = document.getElementById("clinicalAlertCard");
    const alertTitle = document.getElementById("alertTitle");
    const alertDesc = document.getElementById("alertDesc");
    const alertAction = document.getElementById("alertAction");

    if (alertBox && alertData) {
        alertBox.style.borderLeftColor = alertData.risk_color || "var(--status-amber)";
        alertBox.style.background = `${alertData.risk_color}14`;

        if (alertTitle) {
            alertTitle.innerHTML = `⚠️ ${alertData.risk_level} RISK: ${alertData.category}`;
            alertTitle.style.color = alertData.risk_color;
        }
        if (alertDesc) alertDesc.innerText = alertData.clinical_note;
        if (alertAction) alertAction.innerHTML = `<strong>Caregiver Action:</strong> ${alertData.caregiver_guidance}`;
    }
}

function setBar(prefix, pct) {
    const label = document.getElementById(`prob${prefix}`);
    const fill = document.getElementById(`fill${prefix}`);
    if (label) label.innerText = `${pct.toFixed(1)}%`;
    if (fill) fill.style.width = `${pct.toFixed(1)}%`;
}

/* =========================================================
   8. SETTINGS MODAL & HOSTING CONFIGURATION
   ========================================================= */
function initSettingsModal() {
    const modal = document.getElementById("settingsModal");
    const openBtn = document.getElementById("openSettingsBtn");
    const closeBtn = document.getElementById("closeSettingsBtn");
    const saveBtn = document.getElementById("saveApiUrlBtn");
    const input = document.getElementById("customApiInput");

    if (input) input.value = API_BASE;

    if (openBtn && modal) openBtn.onclick = () => modal.classList.add("open");
    if (closeBtn && modal) closeBtn.onclick = () => modal.classList.remove("open");

    if (saveBtn && input && modal) {
        saveBtn.onclick = () => {
            const url = input.value.trim().replace(/\/+$/, "");
            if (url) {
                API_BASE = url;
                localStorage.setItem("multisense_api_url", url);
                modal.classList.remove("open");
                checkBackendHealth();
                alert("API Base URL updated to: " + url);
            }
        };
    }
}

// Global functions for inline HTML event handlers
window.loadDemoVideo = loadDemoVideo;
window.applySensorPreset = applySensorPreset;
window.applyKinematicPreset = applyKinematicPreset;
window.executeMultimodalPrediction = executeMultimodalPrediction;
window.executeKinematicPrediction = executeKinematicPrediction;
window.executeGanGeneration = executeGanGeneration;

/* =========================================================
   9. HARDWARE SENSOR STREAM & MONGODB TELEMETRY
   ========================================================= */

function initHardwareSensorTab() {
    const textarea = document.getElementById("rawSensorTextInput");
    if (textarea) {
        textarea.addEventListener("input", () => {
            updateSensorBadges(textarea.value);
        });
        updateSensorBadges(textarea.value);
    }
}

function updateSensorBadges(text) {
    const badgeIr = document.getElementById("badgeIrStatus");
    const badgeAcc = document.getElementById("badgeAccStatus");
    const badgeLux = document.getElementById("badgeLuxStatus");

    let ir = 0, ax = 0, ay = 0, az = 0, lux = 0;

    const irMatch = text.match(/IR\s*:\s*([+-]?\d+(?:\.\d+)?)/i);
    if (irMatch) ir = parseFloat(irMatch[1]);

    const accMatch = text.match(/ACC\s*:\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)/i);
    if (accMatch) {
        ax = parseFloat(accMatch[1]);
        ay = parseFloat(accMatch[2]);
        az = parseFloat(accMatch[3]);
    }

    const luxMatch = text.match(/Light\s*:\s*([+-]?\d+(?:\.\d+)?)/i);
    if (luxMatch) lux = parseFloat(luxMatch[1]);

    const mag = Math.sqrt(ax * ax + ay * ay + az * az).toFixed(2);
    const calLux = Math.max(0, lux).toFixed(2);

    if (badgeIr) {
        if (ir <= 10) {
            badgeIr.innerHTML = `IR: ${ir} <span style="font-weight:400; font-size:0.72rem; color:var(--text-muted);">(No Skin Contact)</span>`;
            badgeIr.style.color = "#f43f5e";
        } else {
            badgeIr.innerHTML = `IR: ${ir} <span style="font-weight:400; font-size:0.72rem; color:var(--status-green);">(Pulse Active)</span>`;
            badgeIr.style.color = "var(--status-green)";
        }
    }

    if (badgeAcc) {
        const desc = mag > 2.0 ? "High Dynamic Motion" : (mag > 0.3 ? "Subtle Motion" : "Stationary / Calm");
        badgeAcc.innerHTML = `${mag} g <span style="font-weight:400; font-size:0.72rem; color:var(--text-muted);">(${desc})</span>`;
    }

    if (badgeLux) {
        const offsetNotice = lux < 0 ? ` [offset ${lux} clamped to 0]` : "";
        badgeLux.innerHTML = `${calLux} Lux <span style="font-weight:400; font-size:0.72rem; color:var(--text-muted);">${offsetNotice}</span>`;
    }
}

async function checkMongoStatus() {
    const indicator = document.getElementById("mongoStatusIndicator");
    const statusText = document.getElementById("mongoStatusText");
    const pulsingDot = document.getElementById("mongoPulsingDot");
    const dbName = document.getElementById("mongoDbName");
    const collName = document.getElementById("mongoCollName");
    const totalCount = document.getElementById("mongoTotalCount");

    try {
        const res = await fetch(`${API_BASE}/api/db-status`);
        if (!res.ok) throw new Error("DB Status fetch failed");
        const data = await res.json();

        if (dbName) dbName.innerText = data.database || "multisense_db";
        if (collName) collName.innerText = data.collection || "sensor_telemetry";
        if (totalCount) totalCount.innerText = data.total_records || 0;

        if (data.connected) {
            if (indicator) {
                indicator.style.background = "rgba(16, 185, 129, 0.15)";
                indicator.style.color = "var(--status-green)";
                indicator.style.borderColor = "rgba(16, 185, 129, 0.3)";
            }
            if (pulsingDot) pulsingDot.style.background = "var(--status-green)";
            if (statusText) statusText.innerText = "MongoDB Connected";
        } else if (data.has_uri) {
            if (indicator) {
                indicator.style.background = "rgba(239, 68, 68, 0.15)";
                indicator.style.color = "var(--status-red)";
                indicator.style.borderColor = "rgba(239, 68, 68, 0.3)";
            }
            if (pulsingDot) pulsingDot.style.background = "var(--status-red)";
            if (statusText) statusText.innerText = "MongoDB Reconnecting";
        } else {
            if (indicator) {
                indicator.style.background = "rgba(245, 158, 11, 0.15)";
                indicator.style.color = "var(--status-amber)";
                indicator.style.borderColor = "rgba(245, 158, 11, 0.3)";
            }
            if (pulsingDot) pulsingDot.style.background = "var(--status-amber)";
            if (statusText) statusText.innerText = "Buffered Locally (Ready for URI)";
        }
    } catch (e) {
        if (statusText) statusText.innerText = "DB Check Offline";
    }
}

function toggleMongoConfigDrawer() {
    const drawer = document.getElementById("mongoConfigDrawer");
    if (drawer) {
        drawer.style.display = drawer.style.display === "none" || !drawer.style.display ? "flex" : "none";
    }
}

async function saveMongoUri() {
    const input = document.getElementById("mongoUriInput");
    const msg = document.getElementById("mongoConfigMsg");
    if (!input) return;

    const uri = input.value.trim();
    if (!uri) {
        if (msg) {
            msg.innerText = "Please enter a valid MongoDB connection string.";
            msg.style.color = "var(--status-red)";
        }
        return;
    }

    if (msg) {
        msg.innerText = "Testing MongoDB connection...";
        msg.style.color = "var(--primary-cyan)";
    }

    try {
        const res = await fetch(`${API_BASE}/api/db-configure`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mongodb_uri: uri })
        });
        const data = await res.json();
        if (data.connected) {
            if (msg) {
                msg.innerText = `Connected successfully to database '${data.status.database}'!`;
                msg.style.color = "var(--status-green)";
            }
            checkMongoStatus();
            refreshSensorHistory();
            setTimeout(toggleMongoConfigDrawer, 2500);
        } else {
            if (msg) {
                msg.innerText = `Connection failed: ${data.message}`;
                msg.style.color = "var(--status-red)";
            }
        }
    } catch (e) {
        if (msg) {
            msg.innerText = `Error: ${e.message}`;
            msg.style.color = "var(--status-red)";
        }
    }
}

function applyHardwarePreset(preset) {
    const textarea = document.getElementById("rawSensorTextInput");
    if (!textarea) return;

    if (preset === "user_reading") {
        textarea.value = `IR: 0\nACC: 0, 0, 0\nLight: -2.00 lux`;
    } else if (preset === "flapping") {
        textarea.value = `IR: 125\nACC: 1.8, 2.4, 3.2\nLight: 450.00 lux`;
    } else if (preset === "banging") {
        textarea.value = `IR: 148\nACC: 0.8, 4.8, 1.2\nLight: 380.00 lux`;
    } else if (preset === "spinning") {
        textarea.value = `IR: 112\nACC: 1.2, 1.0, 4.1\nLight: 260.00 lux`;
    }
    updateSensorBadges(textarea.value);
}

async function executeSensorStreamInference() {
    const textarea = document.getElementById("rawSensorTextInput");
    const deviceInput = document.getElementById("sensorDeviceId");
    const btn = document.getElementById("btnSendSensorToMlAndDb");
    if (!textarea) return;

    const rawText = textarea.value.trim();
    const deviceId = deviceInput ? deviceInput.value.trim() : "sensor_node_01";

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Processing ML & Storing in DB...`;
    }

    try {
        const res = await fetch(`${API_BASE}/api/sensor-stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                raw_text: rawText,
                device_id: deviceId
            })
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error: ${res.status}`);
        }

        const data = await res.json();

        // Feed prediction into main clinical results card on right panel
        displayPrediction({
            predicted_class: data.ml_output.predicted_class,
            class_id: data.ml_output.class_id,
            confidence: data.ml_output.confidence,
            probabilities: data.ml_output.probabilities,
            model_used: data.ml_output.model_used,
            clinical_alert: data.clinical_alert,
            biomechanical_summary: {
                acc_mag: data.computed_features.acc_magnitude_g,
                calibrated_lux: data.computed_features.calibrated_lux,
                heart_rate_bpm: data.computed_features.heart_rate_bpm
            }
        });

        // Refresh records table
        refreshSensorHistory();
        checkMongoStatus();

    } catch (err) {
        alert("Sensor ingestion error: " + err.message);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `🚀 Ingest Sensors &rarr; Multimodal ML Model &rarr; Save to MongoDB`;
        }
    }
}

async function refreshSensorHistory() {
    const tbody = document.getElementById("sensorHistoryTableBody");
    if (!tbody) return;

    try {
        const res = await fetch(`${API_BASE}/api/sensor-history?limit=15`);
        if (!res.ok) return;
        const data = await res.json();
        const records = data.records || [];

        if (records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" style="padding: 1.5rem; text-align: center; color: var(--text-dim);">No sensor stream records ingested yet. Click 'Ingest Sensors' above to generate first telemetry record!</td></tr>`;
            return;
        }

        tbody.innerHTML = records.map(r => {
            const timeStr = new Date(r.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            const s = r.input_sensors || {};
            const ml = r.ml_output || {};
            const alert = r.clinical_alert || {};
            const isDb = r.storage_status === "SAVED_TO_MONGODB";

            const storageBadge = isDb
                ? `<span style="background: rgba(16, 185, 129, 0.2); color: var(--status-green); padding: 0.15rem 0.45rem; border-radius: 4px; font-weight: 700; font-size: 0.7rem;">MONGODB</span>`
                : `<span style="background: rgba(245, 158, 11, 0.2); color: var(--status-amber); padding: 0.15rem 0.45rem; border-radius: 4px; font-weight: 700; font-size: 0.7rem;">BUFFER</span>`;

            const riskBadge = `<span style="color: ${alert.risk_color || '#fff'}; font-weight: 700;">${alert.risk_level || 'N/A'}</span>`;
            const confPct = ml.confidence ? (ml.confidence * 100).toFixed(1) + "%" : "-";

            return `
                <tr style="border-bottom: 1px solid var(--border-subtle); transition: background 0.15s;">
                    <td style="padding: 0.55rem 0.8rem; font-family: var(--font-mono); color: var(--text-muted);">${timeStr}</td>
                    <td style="padding: 0.55rem 0.8rem; font-family: var(--font-mono);">
                        IR: <span style="color:#f43f5e;">${s.ir !== undefined ? s.ir : 0}</span> | 
                        ACC: <span style="color:var(--primary-cyan);">${r.computed_features ? r.computed_features.acc_magnitude_g + 'g' : '-'}</span> | 
                        Lux: <span style="color:#fbbf24;">${s.calibrated_light_lux !== undefined ? s.calibrated_light_lux : '-'}</span>
                    </td>
                    <td style="padding: 0.55rem 0.8rem; font-weight: 700; color: #fff;">${ml.predicted_class || 'Unknown'}</td>
                    <td style="padding: 0.55rem 0.8rem; font-family: var(--font-mono); color: var(--primary-cyan);">${confPct}</td>
                    <td style="padding: 0.55rem 0.8rem;">${riskBadge}</td>
                    <td style="padding: 0.55rem 0.8rem;">${storageBadge}</td>
                </tr>
            `;
        }).join("");

    } catch (e) {
        console.error("Could not refresh sensor history:", e);
    }
}

window.toggleMongoConfigDrawer = toggleMongoConfigDrawer;
window.saveMongoUri = saveMongoUri;
window.applyHardwarePreset = applyHardwarePreset;
window.executeSensorStreamInference = executeSensorStreamInference;
window.refreshSensorHistory = refreshSensorHistory;
