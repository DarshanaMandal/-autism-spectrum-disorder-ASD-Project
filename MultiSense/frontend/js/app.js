/**
 * MultiSense Frontend Application Logic - Modern Clinical Suite
 * Handles FastAPI REST endpoints, real-time telemetry streaming,
 * microcontroller serial parsing, automated MongoDB sync,
 * video kinematics inference, and diagnostic reporting.
 */

// Determine intelligent API Base URL
let API_BASE = (function getInitialApiBase() {
    const saved = localStorage.getItem("multisense_api_url");
    if (saved && saved.trim().startsWith("http")) {
        return saved.trim().replace(/\/+$/, "");
    }

    const origin = window.location.origin;
    const port = window.location.port;
    const protocol = window.location.protocol;

    // If opened via file:// or common static dev server ports (3000, 5173, 5500, 8080)
    if (!protocol.startsWith("http") || ["3000", "5000", "5173", "5500", "8080"].includes(port)) {
        return "http://127.0.0.1:8000";
    }

    // Default to same origin (e.g. Render, Railway, or FastAPI serving on 8000)
    return origin;
})();

// Real-Time Telemetry & Sync State
let isAutoSyncActive = true;
let syncTimer = null;
let lastKnownRecordId = null;
let isHardwareSimulatorActive = false;
let simulatorTimer = null;

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initHostSettingsModal();
    initHostingGuideModal();
    initSensors();
    initKinematicSliders();
    initVideoUpload();
    initHardwareSensorTab();

    // Initial Health and Data Sync
    checkBackendHealth();
    checkMongoStatus();
    refreshSensorHistory();

    // Start Real-Time Auto-Sync Polling (every 2.5 seconds)
    startAutoSyncLoop();

    // Initialize visualizer pose animator
    if (window.visualizer) {
        window.visualizer.initPoseAnimator();
    }
});

/* =========================================================
   1. BACKEND HEALTH & CONNECTIVITY MONITOR
   ========================================================= */
async function checkBackendHealth() {
    const statusPill = document.getElementById("backendStatusPill");
    const statusText = document.getElementById("backendStatusText");
    const pulsingDot = document.getElementById("backendPulsingDot");
    const swaggerLink = document.getElementById("swaggerDocsLink");

    if (swaggerLink) {
        swaggerLink.href = `${API_BASE}/docs`;
    }

    const startTime = performance.now();
    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
        if (!res.ok) throw new Error("HTTP " + res.status);
        const data = await res.json();
        const latency = Math.round(performance.now() - startTime);

        if (statusText) {
            statusText.innerText = `Online (${latency}ms)`;
        }
        if (pulsingDot) {
            pulsingDot.style.background = "var(--status-green)";
            pulsingDot.style.boxShadow = "0 0 8px var(--status-green)";
        }
        if (statusPill) {
            statusPill.style.borderColor = "rgba(16, 185, 129, 0.4)";
            statusPill.style.color = "var(--status-green)";
            statusPill.title = `Backend Connected: ${API_BASE} (${data.loaded_models ? data.loaded_models.length : 0} models ready)`;
        }
        return true;
    } catch (err) {
        if (statusText) {
            statusText.innerText = "Offline / Reconnecting";
        }
        if (pulsingDot) {
            pulsingDot.style.background = "var(--status-red)";
            pulsingDot.style.boxShadow = "none";
        }
        if (statusPill) {
            statusPill.style.borderColor = "rgba(239, 68, 68, 0.4)";
            statusPill.style.color = "var(--status-red)";
            statusPill.title = `Cannot reach: ${API_BASE}. Click to change backend URL.`;
        }
        return false;
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

            buttons.forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const targetContent = document.getElementById(targetId);
            if (targetContent) targetContent.classList.add("active");

            // Resize canvases on switch
            if (targetId === "tab-telemetry" && window.visualizer) {
                setTimeout(() => window.visualizer.resizeECG(), 50);
            } else if (targetId === "tab-gan" && window.visualizer) {
                setTimeout(() => window.visualizer.resizePose(), 50);
            }
        });
    });
}

/* =========================================================
   3. AUTO-SYNC & LIVE STREAM TELEMETRY ENGINE
   ========================================================= */
let totalInboundPackets = 0;

function startAutoSyncLoop() {
    if (syncTimer) clearInterval(syncTimer);
    syncTimer = setInterval(async () => {
        if (isAutoSyncActive) {
            await pollLiveTelemetry();
        }
        checkBackendHealth();
    }, 1500); // Poll every 1.5s for snappy live stream response
}

async function pollLiveTelemetry() {
    try {
        const res = await fetch(`${API_BASE}/api/sensor-history?limit=1`, { signal: AbortSignal.timeout(2000) });
        if (!res.ok) return;
        const data = await res.json();
        const records = data.records || [];

        if (records.length > 0) {
            const latest = records[0];
            const recId = latest._id || latest.timestamp;

            if (lastKnownRecordId && recId !== lastKnownRecordId) {
                // New incoming telemetry record detected from backend!
                onNewTelemetryStreamReceived(latest);
                refreshSensorHistory();
            }
            lastKnownRecordId = recId;
        }
    } catch (e) {
        // Silently skip transient polling errors
    }
}

function onNewTelemetryStreamReceived(record) {
    totalInboundPackets++;
    const s = record.input_sensors || {};
    const comp = record.computed_features || {};
    const ml = record.ml_output || {};
    const alert = record.clinical_alert || {};
    const timeStr = new Date(record.timestamp || Date.now()).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    // 1. Visual Card Glow Pulse on incoming packet
    const liveCard = document.getElementById("liveInboundCard");
    if (liveCard) {
        liveCard.classList.remove("live-card-pulse");
        void liveCard.offsetWidth;
        liveCard.classList.add("live-card-pulse");
    }

    // 2. Update Feed Header Status
    const feedBadgeText = document.getElementById("inboundFeedText");
    const lastTime = document.getElementById("inboundLastTime");
    const deviceLabel = document.getElementById("inboundDeviceLabel");
    const storageBadge = document.getElementById("inboundStorageBadge");

    if (feedBadgeText) feedBadgeText.innerText = `Stream Active (Packet #${totalInboundPackets})`;
    if (lastTime) lastTime.innerText = `${timeStr} (Live)`;
    if (deviceLabel) deviceLabel.innerText = record.device_id || "arduino_uno";
    if (storageBadge) {
        storageBadge.innerText = record.storage_status === "SAVED_TO_MONGODB" ? "MONGODB CLUSTER" : "MEMORY BUFFER";
        storageBadge.style.color = record.storage_status === "SAVED_TO_MONGODB" ? "var(--status-green)" : "var(--status-amber)";
    }

    // 3. Update live sensor badges
    const badgeIr = document.getElementById("badgeIrStatus");
    const badgeIrSub = document.getElementById("badgeIrSub");
    const badgeAcc = document.getElementById("badgeAccStatus");
    const badgeAccSub = document.getElementById("badgeAccSub");
    const badgeLux = document.getElementById("badgeLuxStatus");
    const badgeLuxSub = document.getElementById("badgeLuxSub");
    const badgeHr = document.getElementById("badgeHrStatus");

    if (badgeIr && s.ir !== undefined) {
        badgeIr.innerText = `IR: ${s.ir}`;
        if (badgeIrSub) badgeIrSub.innerText = s.ir > 10 ? "Optical PPG Engaged" : "Skin Contact: Disengaged";
    }
    if (badgeAcc && comp.acc_magnitude_g !== undefined) {
        badgeAcc.innerText = `${comp.acc_magnitude_g} g`;
        if (badgeAccSub) {
            badgeAccSub.innerText = comp.acc_magnitude_g > 2.5 ? "Severe Kinetic Flutter" : (comp.acc_magnitude_g > 0.8 ? "Motor Agitation" : "Stationary Baseline");
        }
    }
    if (badgeLux && comp.calibrated_lux !== undefined) {
        badgeLux.innerText = `${comp.calibrated_lux} Lux`;
        if (badgeLuxSub) {
            badgeLuxSub.innerText = comp.calibrated_lux > 600 ? "Bright / Photophobia Zone" : (comp.calibrated_lux === 0 ? "Dark / Calibrated" : "Ambient Optimal");
        }
    }
    if (badgeHr && comp.heart_rate_bpm !== undefined) {
        badgeHr.innerText = `${Math.round(comp.heart_rate_bpm)} BPM`;
    }

    // 4. Sync inferred heart rate to ECG canvas
    if (comp.heart_rate_bpm && window.visualizer) {
        window.visualizer.setBPM(comp.heart_rate_bpm);
        const hrDisp = document.getElementById("hrDisplay");
        if (hrDisp) hrDisp.innerText = `${Math.round(comp.heart_rate_bpm)} BPM`;
    }

    // 5. Update Inspector box
    const inspAcc = document.getElementById("inspAccMag");
    const inspLux = document.getElementById("inspLux");
    const inspBuzzer = document.getElementById("inspBuzzer");
    const inspLed = document.getElementById("inspLed");
    const inspCompact = document.getElementById("inspCompactString");
    const inspStorage = document.getElementById("inspectorStorageStatus");

    const isCritical = alert.risk_level === "HIGH" || alert.risk_level === "CRITICAL";
    const isMedium = alert.risk_level === "MEDIUM";

    if (inspAcc) inspAcc.innerText = `${comp.acc_magnitude_g !== undefined ? comp.acc_magnitude_g : 0.0} g`;
    if (inspLux) inspLux.innerText = `${comp.calibrated_lux !== undefined ? comp.calibrated_lux : 0.0} lux`;
    if (inspStorage) {
        inspStorage.innerText = record.storage_status || "SAVED_TO_MONGODB";
        inspStorage.style.color = record.storage_status === "SAVED_TO_MONGODB" ? "var(--status-green)" : "var(--status-amber)";
    }

    if (inspBuzzer) {
        inspBuzzer.innerText = isCritical ? "TRIGGERED (PIN 8 ON)" : "INACTIVE (0)";
        inspBuzzer.style.color = isCritical ? "var(--status-red)" : "var(--status-green)";
    }
    if (inspLed) {
        if (isCritical) {
            inspLed.innerText = "ALERT FLASHING RED (PIN 13)";
            inspLed.style.color = "var(--status-red)";
        } else if (isMedium) {
            inspLed.innerText = "AMBER CAUTION (PIN 13 PULSING)";
            inspLed.style.color = "var(--status-amber)";
        } else {
            inspLed.innerText = "NORMAL GREEN (PIN 13 STEADY)";
            inspLed.style.color = "var(--status-green)";
        }
    }
    if (inspCompact) {
        inspCompact.innerText = `Arduino Stream Response: PRED:${ml.predicted_class || 'Calm'} | CONF:${ml.confidence ? (ml.confidence * 100).toFixed(1) + '%' : '98%'} | RISK:${alert.risk_level || 'LOW'} | BUZZER:${isCritical ? 1 : 0} | DB:${record.storage_status === "SAVED_TO_MONGODB" ? "SAVED" : "BUFFER"}`;
    }

    // 6. Update right-column Diagnostic Hub
    displayPrediction({
        predicted_class: ml.predicted_class,
        class_id: ml.class_id,
        confidence: ml.confidence,
        probabilities: ml.probabilities,
        model_used: ml.model_used || "Multimodal Fusion RF (Live Stream)",
        clinical_alert: alert,
        biomechanical_summary: {
            acc_mag: comp.acc_magnitude_g,
            calibrated_lux: comp.calibrated_lux,
            heart_rate_bpm: comp.heart_rate_bpm
        }
    });
}

function toggleAutoSync() {
    isAutoSyncActive = !isAutoSyncActive;
    const btn = document.getElementById("toggleSyncBtn");
    const icon = document.getElementById("syncBtnIcon");
    const text = document.getElementById("syncBtnText");

    if (isAutoSyncActive) {
        btn.classList.add("active");
        btn.classList.remove("paused");
        if (icon) icon.innerText = "⏸️";
        if (text) text.innerText = "Auto-Sync: ON";
    } else {
        btn.classList.remove("active");
        btn.classList.add("paused");
        if (icon) icon.innerText = "▶️";
        if (text) text.innerText = "Auto-Sync: PAUSED";
    }
}

/* =========================================================
   4. LIVE HARDWARE STREAM SIMULATOR
   ========================================================= */
function toggleLiveHardwareSimulator() {
    const btn = document.getElementById("toggleSimStreamBtn");
    const icon = document.getElementById("simBtnIcon");
    const text = document.getElementById("simBtnText");

    if (isHardwareSimulatorActive) {
        // Stop simulator
        isHardwareSimulatorActive = false;
        clearInterval(simulatorTimer);
        simulatorTimer = null;
        if (btn) {
            btn.classList.remove("running");
            if (icon) icon.innerText = "🎲";
            if (text) text.innerText = "Simulate Live Arduino Stream";
        }
    } else {
        // Start continuous simulator
        isHardwareSimulatorActive = true;
        if (btn) {
            btn.classList.add("running");
            if (icon) icon.innerText = "⏹️";
            if (text) text.innerText = "Stop Live Stream Simulation";
        }

        // Generate dynamic readings every 2 seconds
        let step = 0;
        simulatorTimer = setInterval(async () => {
            step++;
            let irVal, ax, ay, az, luxVal;

            if (step % 6 < 2) {
                // Simulating Arm Flapping burst
                irVal = Math.round(120 + Math.random() * 25);
                ax = (1.8 + Math.random() * 1.5).toFixed(2);
                ay = (2.4 + Math.random() * 1.8).toFixed(2);
                az = (1.5 + Math.random() * 1.2).toFixed(2);
                luxVal = (420 + Math.random() * 40).toFixed(2);
            } else if (step % 6 === 3) {
                // Simulating Sudden SIB Impact
                irVal = Math.round(145 + Math.random() * 20);
                ax = (0.7 + Math.random() * 0.4).toFixed(2);
                ay = (5.1 + Math.random() * 1.2).toFixed(2);
                az = (1.0 + Math.random() * 0.5).toFixed(2);
                luxVal = (350 + Math.random() * 30).toFixed(2);
            } else {
                // Calmer telemetry
                irVal = Math.round(80 + Math.random() * 15);
                ax = (0.2 + Math.random() * 0.3).toFixed(2);
                ay = (0.1 + Math.random() * 0.2).toFixed(2);
                az = (0.9 + Math.random() * 0.2).toFixed(2);
                luxVal = (280 + Math.random() * 30).toFixed(2);
            }

            const rawText = `IR: ${irVal}\nACC: ${ax}, ${ay}, ${az}\nLight: ${luxVal} lux`;
            const textarea = document.getElementById("rawSensorTextInput");
            if (textarea) {
                textarea.value = rawText;
                updateSensorBadges(rawText);
            }

            try {
                const res = await fetch(`${API_BASE}/api/sensor-stream`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        raw_text: rawText,
                        device_id: "arduino_uno_stream_sim"
                    })
                });
                if (res.ok) {
                    const data = await res.json();
                    onNewTelemetryStreamReceived(data);
                    refreshSensorHistory();
                }
            } catch (err) {
                console.warn("Simulator stream error:", err);
            }
        }, 2000);
    }
}

/* =========================================================
   5. HARDWARE SENSOR STREAM TAB & PRESETS
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
    const badgeIrSub = document.getElementById("badgeIrSub");
    const badgeAcc = document.getElementById("badgeAccStatus");
    const badgeAccSub = document.getElementById("badgeAccSub");
    const badgeLux = document.getElementById("badgeLuxStatus");
    const badgeLuxSub = document.getElementById("badgeLuxSub");

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
        badgeIr.innerText = `IR: ${ir}`;
        if (badgeIrSub) badgeIrSub.innerText = ir > 50 ? "Optical PPG Engaged" : "No Skin Contact";
    }
    if (badgeAcc) {
        badgeAcc.innerText = `${mag} g`;
        if (badgeAccSub) {
            badgeAccSub.innerText = mag > 2.5 ? "Severe High Kinetic Velocity" : (mag > 1.2 ? "Moderate Agitation" : "Stationary Baseline");
        }
    }
    if (badgeLux) {
        badgeLux.innerText = `${calLux} Lux`;
        if (badgeLuxSub) {
            badgeLuxSub.innerText = calLux > 500 ? "Bright / Photophobia Zone" : (calLux === "0.00" ? "Dark / Calibrated" : "Optimal Ambient");
        }
    }
}

function applyHardwarePreset(preset) {
    const textarea = document.getElementById("rawSensorTextInput");
    if (!textarea) return;

    if (preset === "user_reading") {
        textarea.value = "IR: 0\nACC: 0, 0, 0\nLight: -2.00 lux";
    } else if (preset === "flapping") {
        textarea.value = "IR: 130\nACC: 2.1, 2.9, 1.8\nLight: 420.00 lux";
    } else if (preset === "banging") {
        textarea.value = "IR: 152\nACC: 0.8, 5.2, 1.0\nLight: 340.00 lux";
    } else if (preset === "spinning") {
        textarea.value = "IR: 114\nACC: 1.2, 1.1, 4.6\nLight: 260.00 lux";
    }
    updateSensorBadges(textarea.value);
}

async function executeSensorStreamInference() {
    const textarea = document.getElementById("rawSensorTextInput");
    const deviceInput = document.getElementById("sensorDeviceId");
    const btn = document.getElementById("btnSendSensorToMlAndDb");
    if (!textarea) return;

    const rawText = textarea.value.trim();
    const deviceId = deviceInput ? deviceInput.value.trim() : "sensor_node_uno_01";

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="pulsing-dot"></span> Processing Multimodal ML & Storing in DB...`;
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

        // Update Right-Hand Diagnostic Hub
        onNewTelemetryStreamReceived(data);

        // Update Backend Inspector Panel
        const inspector = document.getElementById("backendInspector");
        const inspStorage = document.getElementById("inspectorStorageStatus");
        const inspAcc = document.getElementById("inspAccMag");
        const inspLux = document.getElementById("inspLux");
        const inspHr = document.getElementById("inspHr");
        const inspBuzzer = document.getElementById("inspBuzzer");
        const inspCompact = document.getElementById("inspCompactString");

        if (inspector) inspector.style.display = "flex";
        if (inspStorage) {
            inspStorage.innerText = data.storage_status;
            inspStorage.style.color = data.storage_status === "SAVED_TO_MONGODB" ? "var(--status-green)" : "var(--status-amber)";
        }
        if (inspAcc) inspAcc.innerText = data.computed_features.acc_magnitude_g + " g";
        if (inspLux) inspLux.innerText = data.computed_features.calibrated_lux + " lux";
        if (inspHr) inspHr.innerText = data.computed_features.heart_rate_bpm + " bpm";

        const isAlert = data.clinical_alert.risk_level === "HIGH" || data.clinical_alert.risk_level === "CRITICAL";
        if (inspBuzzer) {
            inspBuzzer.innerText = isAlert ? "ACTIVE (PIN 8 BUZZER ON)" : "OFF (0)";
            inspBuzzer.style.color = isAlert ? "var(--status-red)" : "var(--status-green)";
        }
        if (inspCompact) {
            inspCompact.innerText = `Arduino Stream C-String: PRED:${data.ml_output.predicted_class} | CONF:${(data.ml_output.confidence * 100).toFixed(1)}% | RISK:${data.clinical_alert.risk_level} | BUZZER:${isAlert ? 1 : 0} | DB:${data.storage_status === "SAVED_TO_MONGODB" ? "SAVED" : "BUFFER"}`;
        }

        // Refresh records table
        refreshSensorHistory();
        checkMongoStatus();

    } catch (err) {
        alert("Sensor stream ingestion error: " + err.message);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `<span>🚀</span> <span>Send to Multimodal AI Classifier & Save to MongoDB</span>`;
        }
    }
}

/* =========================================================
   6. MONGODB STATUS & TELEMETRY HISTORY
   ========================================================= */
async function checkMongoStatus() {
    const mongoStatusText = document.getElementById("mongoStatusText");
    const mongoPulsingDot = document.getElementById("mongoPulsingDot");
    const navMongoText = document.getElementById("navMongoText");
    const storageMode = document.getElementById("mongoStorageMode");
    const totalCount = document.getElementById("streamRecordCount");

    try {
        const res = await fetch(`${API_BASE}/api/mongo-status`, { signal: AbortSignal.timeout(2500) });
        if (!res.ok) return;
        const data = await res.json();

        if (totalCount) {
            totalCount.innerText = data.total_records || "0";
        }

        if (data.connected) {
            if (mongoStatusText) mongoStatusText.innerText = `MongoDB Connected (${data.total_records} docs)`;
            if (mongoPulsingDot) mongoPulsingDot.style.background = "var(--status-green)";
            if (navMongoText) navMongoText.innerText = `MongoDB: Connected (${data.total_records})`;
            if (storageMode) {
                storageMode.innerText = "MongoDB Atlas";
                storageMode.style.color = "var(--status-green)";
            }
        } else {
            if (mongoStatusText) mongoStatusText.innerText = "Local Memory Buffer (No DB URI)";
            if (mongoPulsingDot) mongoPulsingDot.style.background = "var(--status-amber)";
            if (navMongoText) navMongoText.innerText = "DB: Buffer Mode";
            if (storageMode) {
                storageMode.innerText = "Memory Buffer (150 limit)";
                storageMode.style.color = "var(--status-amber)";
            }
        }
    } catch (e) {
        if (mongoStatusText) mongoStatusText.innerText = "DB Offline";
    }
}

async function refreshSensorHistory() {
    const tbody = document.getElementById("sensorHistoryTableBody");
    const counterBadge = document.getElementById("historyCounterBadge");
    if (!tbody) return;

    try {
        const res = await fetch(`${API_BASE}/api/sensor-history?limit=15`);
        if (!res.ok) return;
        const data = await res.json();
        const records = data.records || [];

        if (counterBadge) {
            counterBadge.innerText = `${records.length} records`;
        }

        if (records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" style="padding: 1.5rem; text-align: center; color: var(--text-dim);">No sensor stream records ingested yet. Click 'Send to Multimodal AI Classifier' or 'Simulate Live Stream' above!</td></tr>`;
            return;
        }

        // On initial page load or refresh, immediately sync Diagnostic Hub to the latest database record!
        if (!lastKnownRecordId && records.length > 0) {
            lastKnownRecordId = records[0]._id || records[0].timestamp;
            onNewTelemetryStreamReceived(records[0]);
        }

        tbody.innerHTML = records.map((r, idx) => {
            const timeStr = new Date(r.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            const s = r.input_sensors || {};
            const comp = r.computed_features || {};
            const ml = r.ml_output || {};
            const alert = r.clinical_alert || {};
            const isDb = r.storage_status === "SAVED_TO_MONGODB";

            const storageBadge = isDb
                ? `<span style="background: rgba(16, 185, 129, 0.2); color: var(--status-green); padding: 0.15rem 0.45rem; border-radius: 4px; font-weight: 700; font-size: 0.7rem;">MONGODB</span>`
                : `<span style="background: rgba(245, 158, 11, 0.2); color: var(--status-amber); padding: 0.15rem 0.45rem; border-radius: 4px; font-weight: 700; font-size: 0.7rem;">BUFFER</span>`;

            const riskBadge = `<span style="color: ${alert.risk_color || '#fff'}; font-weight: 700;">${alert.risk_level || 'N/A'}</span>`;
            const confPct = ml.confidence ? (ml.confidence * 100).toFixed(1) + "%" : "-";

            const flashClass = idx === 0 ? "new-record-flash" : "";

            return `
                <tr class="${flashClass}">
                    <td style="font-family: var(--font-mono); color: var(--text-muted);">${timeStr}</td>
                    <td style="font-family: var(--font-mono);">
                        IR: <span style="color:#f43f5e;">${s.ir !== undefined ? s.ir : 0}</span> | 
                        ACC: <span style="color:var(--primary-cyan);">${comp.acc_magnitude_g !== undefined ? comp.acc_magnitude_g + 'g' : '-'}</span> | 
                        Lux: <span style="color:#fbbf24;">${comp.calibrated_lux !== undefined ? comp.calibrated_lux : '-'}</span>
                    </td>
                    <td style="font-weight: 700; color: #fff;">${ml.predicted_class || 'Unknown'}</td>
                    <td style="font-family: var(--font-mono); color: var(--primary-cyan);">${confPct}</td>
                    <td>${riskBadge}</td>
                    <td>${storageBadge}</td>
                </tr>
            `;
        }).join("");

    } catch (e) {
        console.error("Could not refresh sensor history:", e);
    }
}

function toggleMongoConfigDrawer() {
    const drawer = document.getElementById("mongoConfigDrawer");
    if (!drawer) return;
    drawer.style.display = drawer.style.display === "none" ? "flex" : "none";
}

async function saveMongoUri() {
    const input = document.getElementById("mongoUriInput");
    const msg = document.getElementById("mongoConfigMsg");
    if (!input) return;

    const uri = input.value.trim();
    if (msg) {
        msg.innerText = "Testing MongoDB Atlas connection...";
        msg.style.color = "var(--primary-cyan)";
    }

    try {
        const res = await fetch(`${API_BASE}/api/mongo-config`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ uri: uri })
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

/* =========================================================
   7. VIDEO UPLOAD & MEDIA-PIPE KINEMATICS
   ========================================================= */
function initVideoUpload() {
    const dropZone = document.getElementById("videoDropZone");
    const fileInput = document.getElementById("videoFileInput");
    const analyzeBtn = document.getElementById("analyzeVideoBtn");

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
                handleVideoFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener("change", () => {
            if (fileInput.files && fileInput.files[0]) {
                handleVideoFile(fileInput.files[0]);
            }
        });
    }

    if (analyzeBtn) {
        analyzeBtn.addEventListener("click", () => {
            if (window.selectedVideoBlob) {
                analyzeVideoBlob(window.selectedVideoBlob, window.selectedVideoName || "uploaded_video.mp4");
            } else {
                alert("Please select or drop a video file first!");
            }
        });
    }
}

function handleVideoFile(file) {
    window.selectedVideoBlob = file;
    window.selectedVideoName = file.name;

    const label = document.getElementById("selectedFileName");
    const playerWrapper = document.getElementById("videoPreviewWrapper");
    const video = document.getElementById("videoPlayer");

    if (label) {
        label.innerText = `Selected File: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
        label.style.display = "block";
    }

    if (video && playerWrapper) {
        video.src = URL.createObjectURL(file);
        playerWrapper.style.display = "block";
        video.play().catch(() => {});
    }
}

async function loadDemoVideo(filename) {
    const label = document.getElementById("selectedFileName");
    const playerWrapper = document.getElementById("videoPreviewWrapper");
    const video = document.getElementById("videoPlayer");

    const sampleUrl = `/samples/${filename}`;

    if (label) {
        label.innerText = `Loading verified SSBD clip: ${filename}...`;
        label.style.display = "block";
    }

    try {
        const res = await fetch(sampleUrl);
        if (!res.ok) throw new Error("Sample file not accessible on server");
        const blob = await res.blob();
        window.selectedVideoBlob = blob;
        window.selectedVideoName = filename;

        if (video && playerWrapper) {
            video.src = URL.createObjectURL(blob);
            playerWrapper.style.display = "block";
            video.play().catch(() => {});
        }
        if (label) {
            label.innerText = `Verified SSBD Clip: ${filename} (Click 'Run MediaPipe Extraction' below)`;
        }
    } catch (err) {
        alert(`Could not load ${filename}: ${err.message}`);
    }
}

async function analyzeVideoBlob(blob, filename) {
    const btn = document.getElementById("analyzeVideoBtn");
    const modelChoice = document.getElementById("videoModelChoice") ? document.getElementById("videoModelChoice").value : "gan_augmented";
    const kinGrid = document.getElementById("videoKinematicsGrid");

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="pulsing-dot"></span> Extracting 33 MediaPipe Skeletal Landmarks...`;
    }

    const formData = new FormData();
    formData.append("video", blob, filename);

    try {
        const res = await fetch(`${API_BASE}/predict/video?model_choice=${modelChoice}`, {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Server error: ${res.status}`);
        }

        const data = await res.json();

        // Populate video kinematics grid
        if (kinGrid) {
            kinGrid.style.display = "grid";
            const k = data.extracted_kinematics || {};
            const f1 = document.getElementById("vidFramesExtracted");
            const f2 = document.getElementById("vidWristVel");
            const f3 = document.getElementById("vidHeadPitch");
            const f4 = document.getElementById("vidMovementIntensity");

            if (f1) f1.innerText = `${data.frames_analyzed || 60} frames`;
            if (f2) f2.innerText = k.wrist_velocity_mean ? `${k.wrist_velocity_mean.toFixed(2)} m/s` : "0.52 m/s";
            if (f3) f3.innerText = k.head_pitch_variance ? `${k.head_pitch_variance.toFixed(3)} rad²` : "0.03 rad²";
            if (f4) f4.innerText = k.movement_intensity ? k.movement_intensity.toFixed(2) : "0.45";
        }

        displayPrediction(data);

    } catch (err) {
        alert("Video kinematic extraction error: " + err.message);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `<span>📹</span> <span>Run Full MediaPipe Pose Kinematic Extraction</span>`;
        }
    }
}

/* =========================================================
   8. MULTIMODAL SENSOR SUITE TAB
   ========================================================= */
function initSensors() {
    const hrSlider = document.getElementById("hrSlider");
    const spo2Slider = document.getElementById("spo2Slider");
    const accSlider = document.getElementById("accSlider");
    const luxSlider = document.getElementById("luxSlider");
    const gyroSlider = document.getElementById("gyroSlider");

    if (hrSlider) {
        hrSlider.addEventListener("input", (e) => {
            const bpm = parseInt(e.target.value);
            const hrDisp = document.getElementById("hrDisplay");
            if (hrDisp) hrDisp.innerText = `${bpm} BPM`;
            if (window.visualizer) window.visualizer.setBPM(bpm);
        });
    }

    if (spo2Slider) {
        spo2Slider.addEventListener("input", (e) => {
            const disp = document.getElementById("spo2Display");
            if (disp) disp.innerText = `${parseFloat(e.target.value).toFixed(1)}%`;
        });
    }

    if (accSlider) {
        accSlider.addEventListener("input", (e) => {
            const disp = document.getElementById("accDisplay");
            if (disp) disp.innerText = `${parseFloat(e.target.value).toFixed(1)} g`;
        });
    }

    if (luxSlider) {
        luxSlider.addEventListener("input", (e) => {
            const lux = parseInt(e.target.value);
            const disp = document.getElementById("luxDisplay");
            const warn = document.getElementById("luxWarningBadge");
            if (disp) disp.innerText = `${lux} Lux`;
            if (warn) warn.style.display = lux >= 750 ? "inline-block" : "none";
        });
    }

    if (gyroSlider) {
        gyroSlider.addEventListener("input", (e) => {
            const disp = document.getElementById("gyroDisplay");
            if (disp) disp.innerText = `${parseFloat(e.target.value).toFixed(2)} rad/s`;
        });
    }
}

function applySensorPreset(preset) {
    const hr = document.getElementById("hrSlider");
    const spo2 = document.getElementById("spo2Slider");
    const acc = document.getElementById("accSlider");
    const lux = document.getElementById("luxSlider");
    const gyro = document.getElementById("gyroSlider");

    if (preset === "calm") {
        if (hr) hr.value = 75;
        if (spo2) spo2.value = 99.0;
        if (acc) acc.value = 0.6;
        if (lux) lux.value = 250;
        if (gyro) gyro.value = 0.05;
    } else if (preset === "flapping") {
        if (hr) hr.value = 135;
        if (spo2) spo2.value = 96.5;
        if (acc) acc.value = 3.8;
        if (lux) lux.value = 450;
        if (gyro) gyro.value = 0.45;
    } else if (preset === "meltdown") {
        if (hr) hr.value = 168;
        if (spo2) spo2.value = 92.5;
        if (acc) acc.value = 5.4;
        if (lux) lux.value = 850;
        if (gyro) gyro.value = 2.40;
    }

    [hr, spo2, acc, lux, gyro].forEach(slider => {
        if (slider) slider.dispatchEvent(new Event("input"));
    });

    executeMultimodalPrediction();
}

async function executeMultimodalPrediction() {
    const hr = parseFloat(document.getElementById("hrSlider")?.value || 80);
    const spo2 = parseFloat(document.getElementById("spo2Slider")?.value || 98);
    const acc = parseFloat(document.getElementById("accSlider")?.value || 1.2);
    const lux = parseFloat(document.getElementById("luxSlider")?.value || 350);
    const gyro = parseFloat(document.getElementById("gyroSlider")?.value || 0.2);

    const payload = {
        wrist_velocity_mean: acc > 2.0 ? 0.65 : 0.22,
        wrist_velocity_max: acc * 0.4,
        wrist_velocity_std: 0.15,
        wrist_acceleration_mean: acc * 0.25,
        arm_velocity_mean: acc * 0.35,
        head_velocity_mean: 0.15,
        head_velocity_max: 0.35,
        head_pitch_variance: 0.04,
        shoulder_velocity_mean: 0.18,
        body_velocity_mean: 0.20,
        angular_velocity_centroid: gyro,
        movement_intensity: acc / 6.0,
        pose_variance: 0.04,
        elbow_angle_mean: 110.0,
        elbow_angle_std: 15.0,
        shoulder_angle_mean: 72.0,
        shoulder_angle_std: 12.0,
        max30102_hr_bpm: hr,
        max30102_spo2_pct: spo2,
        mpu6050_acc_mag_mean: acc,
        mpu6050_gyro_yaw_rads: gyro,
        bh1750_lux_mean: lux
    };

    try {
        const res = await fetch(`${API_BASE}/predict/multimodal`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error("Multimodal inference failed");
        const data = await res.json();
        displayPrediction(data);
    } catch (e) {
        console.error("Multimodal predict error:", e);
    }
}

/* =========================================================
   9. KINEMATIC LAB TAB
   ========================================================= */
function initKinematicSliders() {
    const sliders = ["kinWristVel", "kinHeadPitch", "kinHeadVel", "kinAngularVel"];
    sliders.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener("input", () => {
                const disp = document.getElementById(id + "Disp");
                if (disp) {
                    const unit = id.includes("Pitch") ? " rad²" : (id.includes("Angular") ? " rad/s" : " m/s");
                    disp.innerText = `${parseFloat(el.value).toFixed(2)}${unit}`;
                }
                const liveToggle = document.getElementById("livePredictToggle");
                if (liveToggle && liveToggle.checked) {
                    debounceKinematicPredict();
                }
            });
        }
    });
}

let kinDebounceTimer = null;
function debounceKinematicPredict() {
    clearTimeout(kinDebounceTimer);
    kinDebounceTimer = setTimeout(executeKinematicPrediction, 250);
}

function applyKinematicPreset(behavior) {
    const wrist = document.getElementById("kinWristVel");
    const pitch = document.getElementById("kinHeadPitch");
    const headVel = document.getElementById("kinHeadVel");
    const angVel = document.getElementById("kinAngularVel");

    if (behavior === "ArmFlapping") {
        if (wrist) wrist.value = 0.85;
        if (pitch) pitch.value = 0.02;
        if (headVel) headVel.value = 0.10;
        if (angVel) angVel.value = 0.05;
    } else if (behavior === "HeadBanging") {
        if (wrist) wrist.value = 0.18;
        if (pitch) pitch.value = 0.55;
        if (headVel) headVel.value = 0.85;
        if (angVel) angVel.value = 0.08;
    } else if (behavior === "Spinning") {
        if (wrist) wrist.value = 0.35;
        if (pitch) pitch.value = 0.04;
        if (headVel) headVel.value = 0.25;
        if (angVel) angVel.value = 1.45;
    }

    [wrist, pitch, headVel, angVel].forEach(s => s && s.dispatchEvent(new Event("input")));
}

async function executeKinematicPrediction() {
    const wrist = parseFloat(document.getElementById("kinWristVel")?.value || 0.52);
    const pitch = parseFloat(document.getElementById("kinHeadPitch")?.value || 0.03);
    const headVel = parseFloat(document.getElementById("kinHeadVel")?.value || 0.12);
    const angVel = parseFloat(document.getElementById("kinAngularVel")?.value || 0.08);
    const model = document.getElementById("kinModelChoice")?.value || "gan_augmented";

    const payload = {
        wrist_velocity_mean: wrist,
        wrist_velocity_max: wrist * 2.2,
        wrist_velocity_std: wrist * 0.4,
        wrist_acceleration_mean: wrist * 0.6,
        arm_velocity_mean: wrist * 0.75,
        head_velocity_mean: headVel,
        head_velocity_max: headVel * 2.1,
        head_pitch_variance: pitch,
        shoulder_velocity_mean: 0.14,
        body_velocity_mean: 0.15,
        angular_velocity_centroid: angVel,
        movement_intensity: (wrist + headVel + angVel) / 3.0,
        pose_variance: 0.035,
        elbow_angle_mean: 105.0,
        elbow_angle_std: 18.0,
        shoulder_angle_mean: 70.0,
        shoulder_angle_std: 14.0,
        model_choice: model
    };

    try {
        const res = await fetch(`${API_BASE}/predict/features`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error("Kinematic predict failed");
        const data = await res.json();
        displayPrediction(data);
    } catch (e) {
        console.error("Kinematic error:", e);
    }
}

/* =========================================================
   10. TIMEGAN SYNTHETIC GENERATION
   ========================================================= */
async function executeGanGeneration() {
    const btn = document.getElementById("generateGanBtn");
    const classSelect = document.getElementById("ganClassSelect")?.value || "ArmFlapping";
    const countInput = parseInt(document.getElementById("ganCountInput")?.value || "5");
    const resultBox = document.getElementById("ganResultInfo");

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="pulsing-dot"></span> Generating via Bi-GRU TimeGAN...`;
    }

    try {
        const res = await fetch(`${API_BASE}/generate/synthetic`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                behavior_class: classSelect,
                num_samples: countInput,
                noise_seed: Math.floor(Math.random() * 1000)
            })
        });

        if (!res.ok) throw new Error("TimeGAN generation failed");
        const data = await res.json();

        if (resultBox) {
            resultBox.style.display = "block";
            resultBox.innerHTML = `
                <strong>Synthesized Dimension:</strong> <code>[${data.sequence_shape.join(", ")}]</code> &bull; 
                <strong>Class:</strong> <span style="color:var(--primary-cyan); font-weight:700;">${data.behavior_class}</span><br>
                <span>${data.info}</span>
            `;
        }

        // Animate generated frames on pose visualizer canvas
        if (window.visualizer && data.synthetic_sequences) {
            window.visualizer.loadPoseFrames(data.synthetic_sequences[0], data.behavior_class);
        }

    } catch (e) {
        alert("TimeGAN Error: " + e.message);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `<span>🧬</span> <span>Synthesize Biomechanical Trajectories</span>`;
        }
    }
}

/* =========================================================
   11. DIAGNOSTIC HUB RENDERER
   ========================================================= */
function displayPrediction(data) {
    const badge = document.getElementById("diagnosticBadge");
    const conf = document.getElementById("confidenceReadout");
    const modelTag = document.getElementById("activeModelTag");
    const hubCard = document.getElementById("diagnosticHubCard");

    const predClass = data.predicted_class || "ArmFlapping";
    const confidencePct = (data.confidence || 0.85) * 100;

    if (badge) {
        badge.innerText = predClass;
        badge.className = "classification-badge";
        if (predClass === "ArmFlapping") badge.classList.add("badge-arm");
        else if (predClass === "HeadBanging") badge.classList.add("badge-head");
        else if (predClass === "Spinning") badge.classList.add("badge-spin");
        else badge.classList.add("badge-calm");
    }

    if (conf) conf.innerText = `${confidencePct.toFixed(1)}%`;
    if (modelTag) modelTag.innerText = data.model_used || "GAN-Augmented RF";

    // Update probabilities
    const probs = data.probabilities || { ArmFlapping: 0.85, HeadBanging: 0.10, Spinning: 0.05 };
    setBar("Arm", (probs.ArmFlapping || 0) * 100);
    setBar("Head", (probs.HeadBanging || 0) * 100);
    setBar("Spin", (probs.Spinning || 0) * 100);

    // Update Clinical Alert
    const alert = data.clinical_alert || {};
    const alertCard = document.getElementById("clinicalAlertCard");
    const alertTitle = document.getElementById("alertTitle");
    const alertDesc = document.getElementById("alertDesc");
    const alertAction = document.getElementById("alertAction");

    const riskLevel = alert.risk_level || "MEDIUM";
    const riskColor = alert.risk_color || "var(--status-amber)";

    if (alertTitle) {
        alertTitle.innerHTML = `<span style="color:${riskColor};">⚠️ ${riskLevel} RISK:</span> ${alert.category || 'Motor Agitation & Sensory Overflow'}`;
    }
    if (alertDesc) {
        alertDesc.innerText = alert.clinical_implication || "Rapid motor flutter detected in extremities.";
    }
    if (alertAction) {
        alertAction.innerHTML = `<strong>Caregiver Action:</strong> ${alert.caregiver_guidance || 'Offer sensory decompression tools.'}`;
    }
    if (alertCard) {
        alertCard.style.borderLeftColor = riskColor;
        alertCard.style.background = `rgba(${riskLevel === 'HIGH' ? '239, 68, 68' : '245, 158, 11'}, 0.08)`;
    }

    // Glow effect around the diagnostic hub card on high risk
    if (hubCard) {
        if (riskLevel === "HIGH" || riskLevel === "CRITICAL") {
            hubCard.style.boxShadow = "0 0 35px rgba(239, 68, 68, 0.4)";
            hubCard.style.borderColor = "rgba(239, 68, 68, 0.6)";
        } else if (riskLevel === "MEDIUM") {
            hubCard.style.boxShadow = "0 0 20px rgba(245, 158, 11, 0.25)";
            hubCard.style.borderColor = "rgba(245, 158, 11, 0.4)";
        } else {
            hubCard.style.boxShadow = "var(--shadow-card)";
            hubCard.style.borderColor = "var(--border-subtle)";
        }
    }

    // Synchronize Hardware Actuator Feedback Box (Buzzer & Pin 13 LED)
    const buzzerStatus = document.getElementById("hubBuzzerStatus");
    const ledStatus = document.getElementById("hubLedStatus");
    const isCritical = riskLevel === "HIGH" || riskLevel === "CRITICAL";
    const isMedium = riskLevel === "MEDIUM";

    if (buzzerStatus) {
        buzzerStatus.innerText = isCritical ? "TRIGGERED (PIN 8 BUZZER ON)" : "INACTIVE (0)";
        buzzerStatus.style.color = isCritical ? "var(--status-red)" : "var(--status-green)";
    }
    if (ledStatus) {
        if (isCritical) {
            ledStatus.innerText = "ALERT FLASHING RED (PIN 13)";
            ledStatus.style.color = "var(--status-red)";
        } else if (isMedium) {
            ledStatus.innerText = "AMBER CAUTION (PIN 13 PULSING)";
            ledStatus.style.color = "var(--status-amber)";
        } else {
            ledStatus.innerText = "NORMAL GREEN (PIN 13 STEADY)";
            ledStatus.style.color = "var(--status-green)";
        }
    }
}

function setBar(prefix, pct) {
    const label = document.getElementById(`prob${prefix}`);
    const fill = document.getElementById(`fill${prefix}`);
    if (label) label.innerText = `${pct.toFixed(1)}%`;
    if (fill) fill.style.width = `${pct.toFixed(1)}%`;
}

/* =========================================================
   12. HOST SETTINGS & HOSTING GUIDE MODALS
   ========================================================= */
function initHostSettingsModal() {
    const modal = document.getElementById("settingsModal");
    const openBtn = document.getElementById("openSettingsBtn");
    const closeBtn = document.getElementById("closeSettingsBtn");
    const saveBtn = document.getElementById("saveApiUrlBtn");
    const input = document.getElementById("customApiInput");

    if (input) input.value = API_BASE;

    if (openBtn && modal) openBtn.onclick = () => {
        if (input) input.value = API_BASE;
        modal.classList.add("open");
    };
    if (closeBtn && modal) closeBtn.onclick = () => modal.classList.remove("open");

    if (saveBtn && input && modal) {
        saveBtn.onclick = () => {
            const url = input.value.trim().replace(/\/+$/, "");
            if (url) {
                API_BASE = url;
                localStorage.setItem("multisense_api_url", url);
                modal.classList.remove("open");
                checkBackendHealth();
                refreshSensorHistory();
            }
        };
    }
}

function openHostSettings() {
    const modal = document.getElementById("settingsModal");
    const input = document.getElementById("customApiInput");
    if (input) input.value = API_BASE;
    if (modal) modal.classList.add("open");
}

function setApiPreset(url) {
    const input = document.getElementById("customApiInput");
    if (input) input.value = url;
    testCurrentEndpoint();
}

async function testCurrentEndpoint() {
    const input = document.getElementById("customApiInput");
    const result = document.getElementById("connectionTestResult");
    if (!input || !result) return;

    const url = input.value.trim().replace(/\/+$/, "");
    result.innerText = `Testing connection to ${url}/health ...`;
    result.style.color = "var(--primary-cyan)";

    const start = performance.now();
    try {
        const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(3000) });
        if (!res.ok) throw new Error("HTTP " + res.status);
        const data = await res.json();
        const latency = Math.round(performance.now() - start);

        result.innerHTML = `✅ Connected! Latency: ${latency}ms &bull; Models: ${data.loaded_models ? data.loaded_models.length : 0} ready`;
        result.style.color = "var(--status-green)";
    } catch (e) {
        result.innerHTML = `❌ Connection Failed: ${e.message}. Ensure backend is running.`;
        result.style.color = "var(--status-red)";
    }
}

function initHostingGuideModal() {
    const modal = document.getElementById("hostingGuideModal");
    const openBtn = document.getElementById("openHostingGuideBtn");
    const closeBtn = document.getElementById("closeHostingGuideBtn");

    if (openBtn && modal) openBtn.onclick = () => modal.classList.add("open");
    if (closeBtn && modal) closeBtn.onclick = () => modal.classList.remove("open");
}

function closeHostingGuide() {
    const modal = document.getElementById("hostingGuideModal");
    if (modal) modal.classList.remove("open");
}

// Global functions for inline HTML event handlers
window.openHostSettings = openHostSettings;
window.setApiPreset = setApiPreset;
window.testCurrentEndpoint = testCurrentEndpoint;
window.closeHostingGuide = closeHostingGuide;
window.toggleAutoSync = toggleAutoSync;
window.toggleLiveHardwareSimulator = toggleLiveHardwareSimulator;
window.toggleMongoConfigDrawer = toggleMongoConfigDrawer;
window.saveMongoUri = saveMongoUri;
window.applyHardwarePreset = applyHardwarePreset;
window.executeSensorStreamInference = executeSensorStreamInference;
window.refreshSensorHistory = refreshSensorHistory;
window.loadDemoVideo = loadDemoVideo;
window.applySensorPreset = applySensorPreset;
window.applyKinematicPreset = applyKinematicPreset;
window.executeMultimodalPrediction = executeMultimodalPrediction;
window.executeKinematicPrediction = executeKinematicPrediction;
window.executeGanGeneration = executeGanGeneration;
