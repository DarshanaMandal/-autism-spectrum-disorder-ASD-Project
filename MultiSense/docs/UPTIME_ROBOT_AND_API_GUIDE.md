# MultiSense: UptimeRobot & API Endpoints Reference Guide

**Live Deployment URL:** `https://multisense-ai-pslv.onrender.com/`  
**Swagger OpenAPI Documentation:** `https://multisense-ai-pslv.onrender.com/docs`  

---

## 1. UptimeRobot Configuration (Keep Render Awake 24/7)

Render's free tier automatically suspends web services after **15 minutes** of inactivity (causing 30-50s cold start delays). UptimeRobot sends periodic pings to keep the server hot and responsive.

### Recommended Settings in UptimeRobot:
- **Monitor Type:** `HTTP(s)`
- **Friendly Name:** `MultiSense Clinical AI KeepAlive`
- **URL (or IP):** `https://multisense-ai-pslv.onrender.com/ping`  *(or `/health`)*
- **Monitoring Interval:** `Every 5 minutes` *(Render sleeps at 15 mins, so 5 mins prevents sleep)*
- **HTTP Method:** `HEAD` or `GET` *(Both are supported with 0ms overhead)*
- **Accepted HTTP Statuses:** `200`
- **Alert Contacts:** Your email

> [!TIP]
> Both `GET` and `HEAD` requests to `/ping` and `/health` return immediate `200 OK` with `Cache-Control: no-cache, no-store, must-revalidate` to prevent cached responses and ensure the container stays active.

---

## 2. All Available Routes & Endpoints

| Category | Method | Route | Description |
|---|---|---|---|
| **User Interface** | `GET` / `HEAD` | `/` | Clinical Web Dashboard (Single Page App) |
| **User Interface** | `GET` / `HEAD` | `/dashboard` | Alternative Dashboard URL |
| **Keep-Alive (UptimeRobot)** | `GET` / `HEAD` | `/ping` or `/api/ping` | Dedicated 0ms Keep-Alive Endpoint |
| **System Health** | `GET` / `HEAD` | `/health` | Health status & loaded model registry |
| **System Info** | `GET` | `/api/info` | API metadata & supported behaviors |
| **Models Registry** | `GET` | `/models` | Trained models stats (84.65%, 87.98%, 97.44%) |
| **Hardware Stream** | `POST` | `/api/sensor-stream` | Ingests raw serial text or JSON & saves to MongoDB |
| **Arduino Stream** | `GET` / `POST` | `/api/arduino/stream` | Ultra-lightweight endpoint for Arduino Uno / ESP32 |
| **MongoDB History** | `GET` | `/api/sensor-history` | Returns recent telemetry audit logs from MongoDB |
| **MongoDB Status** | `GET` | `/api/mongo-status` | Checks MongoDB Atlas connection & doc counts |
| **MongoDB Config** | `POST` | `/api/mongo-config` | Dynamically update MongoDB Atlas URI |
| **Video Kinematics** | `POST` | `/predict/video` | Extracts 33 MediaPipe pose landmarks from MP4/AVI |
| **Kinematic Lab** | `POST` | `/predict/features` | 17-feature Kinematic Random Forest inference |
| **Multimodal Fusion**| `POST` | `/predict/multimodal`| 22-feature Video + PPG + IMU + Lux classifier |
| **TimeGAN Generator** | `POST` | `/generate/synthetic` | Conditional Bi-GRU synthetic sequence generation |
| **API Docs** | `GET` | `/docs` | Interactive Swagger UI |
| **ReDoc** | `GET` | `/redoc` | Interactive ReDoc UI |

---

## 3. Detailed Request & Response Formats

### A. Keep-Alive / UptimeRobot Route
- **URL:** `https://multisense-ai-pslv.onrender.com/ping`
- **Method:** `GET` or `HEAD`
- **Response (200 OK):**
```json
{
  "status": "ok",
  "uptime_robot": "active",
  "service": "multisense-clinical-ai",
  "timestamp": "2026-10-16T16:08:02.216997+00:00"
}
```

---

### B. System Health Route
- **URL:** `https://multisense-ai-pslv.onrender.com/health`
- **Method:** `GET` or `HEAD`
- **Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cpu",
  "loaded_models": [
    "Baseline RF",
    "GAN-Augmented RF",
    "Multimodal Sensor Fusion RF",
    "Conditional Bi-GRU GAN Generator"
  ],
  "supported_classes": ["ArmFlapping", "HeadBanging", "Spinning"]
}
```

---

### C. Hardware Serial Stream & Database Ingestion Route
- **URL:** `https://multisense-ai-pslv.onrender.com/api/sensor-stream`
- **Method:** `POST`
- **Request Headers:** `Content-Type: application/json`
- **Request Payload (Option 1 - Raw Microcontroller Text):**
```json
{
  "raw_text": "IR: 130\nACC: 2.1, 2.9, 1.8\nLight: 420.00 lux",
  "device_id": "wearable_node_uno_01"
}
```
- **Request Payload (Option 2 - Structured JSON):**
```json
{
  "ir": 130.0,
  "acc_x": 2.1,
  "acc_y": 2.9,
  "acc_z": 1.8,
  "light_lux": 420.0,
  "device_id": "wearable_node_uno_01"
}
```
- **Response (200 OK):**
```json
{
  "success": true,
  "record_id": "efa9e7db-36e8-491e-87ed-d6f5843dab71",
  "timestamp": "2026-10-16T16:08:15.377312+00:00",
  "device_id": "wearable_node_uno_01",
  "storage_status": "SAVED_TO_MONGODB",
  "input_sensors": {
    "raw_text": "IR: 130\nACC: 2.1, 2.9, 1.8\nLight: 420.00 lux",
    "ir": 130.0,
    "acc_x": 2.1,
    "acc_y": 2.9,
    "acc_z": 1.8,
    "light_lux": 420.0,
    "calibrated_light_lux": 420.0
  },
  "computed_features": {
    "acc_magnitude_g": 3.99,
    "calibrated_lux": 420.0,
    "heart_rate_bpm": 130.0,
    "spo2_pct": 98.0,
    "skin_contact_detected": true,
    "gyro_yaw_rads": 0.45
  },
  "ml_output": {
    "predicted_class": "ArmFlapping",
    "class_id": 0,
    "confidence": 0.9645,
    "probabilities": {
      "ArmFlapping": 0.9645,
      "HeadBanging": 0.0041,
      "Spinning": 0.0314
    },
    "model_used": "Multimodal Sensor Fusion RF (Accuracy: 97.44%)"
  },
  "clinical_alert": {
    "risk_level": "MEDIUM",
    "risk_color": "#f59e0b",
    "category": "Motor Agitation & Sensory Overflow",
    "clinical_note": "Rapid bilateral wrist/arm flutter detected.",
    "caregiver_guidance": "Provide sensory decompression tools (weighted lap pad, fidgets)."
  }
}
```

---

### D. Dedicated Lightweight Arduino Uno Endpoint
- **URL:** `https://multisense-ai-pslv.onrender.com/api/arduino/stream?ir=0&ax=0&ay=0&az=0&lux=-2.0`
- **Method:** `GET` or `POST`
- **Response (Compact JSON for microcontrollers with 2KB RAM):**
```json
{
  "ok": true,
  "pred": "ArmFlapping",
  "conf": 0.4533,
  "risk": "MEDIUM",
  "buzzer": 0,
  "db": "SAVED",
  "rec_id": "d37aaeef"
}
```
- **Response (Plain text compact format for serial parsers `format=text`):**
```
PRED:ArmFlapping|CONF:0.45|RISK:MEDIUM|BUZZER:0|DB:SAVED
```

---

### E. MongoDB Telemetry History Route
- **URL:** `https://multisense-ai-pslv.onrender.com/api/sensor-history?limit=15`
- **Method:** `GET`
- **Response (200 OK):**
```json
{
  "total_returned": 1,
  "storage_source": "MONGODB",
  "records": [
    {
      "_id": "efa9e7db-36e8-491e-87ed-d6f5843dab71",
      "timestamp": "2026-10-16T16:08:15.377312+00:00",
      "device_id": "wearable_node_uno_01",
      "storage_status": "SAVED_TO_MONGODB",
      "input_sensors": { ... },
      "computed_features": { ... },
      "ml_output": { ... },
      "clinical_alert": { ... }
    }
  ]
}
```
