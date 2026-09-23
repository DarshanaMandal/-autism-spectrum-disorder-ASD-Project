# 📟 MultiSense Arduino Uno Integration Guide

This guide explains how to connect your **Arduino Uno**, read your 3 sensors (`IR`, `ACC`, `Light`), send telemetry to the **MultiSense ML Model**, and persist every reading into **MongoDB Atlas**.

---

## 🚀 1. Available API Endpoints for Arduino

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **`POST`** | `/api/arduino/stream?format=text` | **(Recommended for Arduino)** Ingests sensor text & returns clean pipe-delimited text |
| **`POST`** | `/api/arduino/stream?format=json` | Ingests sensor text or JSON & returns compact JSON |
| **`GET`** | `/api/arduino/stream?ir=0&ax=0&ay=0&az=0&lux=-2.0&format=text` | Query param ingestion (useful for simple HTTP clients) |
| **`POST`** | `/api/sensor-stream` | Full clinical ingestion endpoint (returns complete JSON + clinical explanation) |

---

## 📡 2. Request & Response Specifications

### Format A: Plain Text (Ideal for Arduino Uno - Low RAM)

#### ➡️ Request
- **URL**: `http://<SERVER_IP>:8000/api/arduino/stream?format=text`
- **Method**: `POST`
- **Header**: `Content-Type: text/plain`
- **Body**:
```text
IR: 0
ACC: 0, 0, 0
Light: -2.00 lux
```

#### ⬅️ Response
- **Status**: `200 OK`
- **Content-Type**: `text/plain`
- **Body**:
```text
PRED:ArmFlapping|CONF:0.45|RISK:MEDIUM|BUZZER:0|DB:SAVED
```
> **Field Explanation:**
> - `PRED`: Predicted stimming behavior (`ArmFlapping`, `HeadBanging`, or `Spinning`)
> - `CONF`: ML confidence score (e.g. `0.45`)
> - `RISK`: Clinical Risk Alert (`LOW`, `MEDIUM`, `HIGH`)
> - `BUZZER`: `1` if HIGH risk alert (triggers physical buzzer/LED on Arduino), `0` otherwise
> - `DB`: `SAVED` if stored directly in MongoDB Atlas, `BUFFERED` if buffered locally

---

### Format B: Compact JSON

#### ➡️ Request
- **URL**: `http://<SERVER_IP>:8000/api/arduino/stream`
- **Method**: `POST`
- **Header**: `Content-Type: application/json`
- **Body**:
```json
{
  "ir": 0,
  "acc_x": 0.0,
  "acc_y": 0.0,
  "acc_z": 0.0,
  "light_lux": -2.0,
  "device_id": "arduino_uno_node_01"
}
```

#### ⬅️ Response
- **Status**: `200 OK`
- **Body**:
```json
{
  "ok": true,
  "pred": "ArmFlapping",
  "conf": 0.4533,
  "risk": "MEDIUM",
  "buzzer": 0,
  "db": "SAVED",
  "rec_id": "da4ae478"
}
```

---

### Format C: HTTP GET with Query Parameters

#### ➡️ Request
```text
GET /api/arduino/stream?ir=0&ax=0&ay=0&az=0&lux=-2.0&format=text
```

#### ⬅️ Response
```text
PRED:ArmFlapping|CONF:0.45|RISK:MEDIUM|BUZZER:0|DB:SAVED
```

---

## 🔌 3. Hardware Setup & Pin Connections

### Arduino Uno Pinout:
| Sensor / Component | Arduino Uno Pin | Protocol |
| :--- | :--- | :--- |
| **MPU6050 SDA** | **A4** (or dedicated SDA) | I2C |
| **MPU6050 SCL** | **A5** (or dedicated SCL) | I2C |
| **MPU6050 VCC / GND** | 5V (or 3.3V) / GND | Power |
| **MAX30102 / IR Sensor** | Pin **A0** (or I2C SDA/SCL) | Analog / I2C |
| **BH1750 / Light Sensor** | Pin **A1** (or I2C SDA/SCL) | Analog / I2C |
| **Buzzer (Active)** | Pin **8** | Digital Output |
| **Status Alert LED** | Pin **13** (Built-in LED) | Digital Output |

---

## 🛠️ 4. How to Run (2 Simple Ways)

### Method 1: USB Cable to PC/Laptop (Standard & Easiest for Uno)
Since standard Arduino Uno does not have built-in WiFi, you connect it to your computer via USB cable.

1. Flash the sketch [`arduino_uno_sensors.ino`](file:///Users/shreyashgautam/Desktop/darshana/authism/MultiSense/hardware/arduino_uno_sensors/arduino_uno_sensors.ino) to your Arduino Uno using Arduino IDE.
2. Run the Python Serial Bridge on your PC/Mac:
```bash
python3 MultiSense/hardware/serial_bridge.py
```
> The bridge automatically detects your Arduino's USB port, reads `IR: 0 \n ACC: 0, 0, 0 \n Light: -2.00 lux`, calls the MultiSense ML API, stores it in MongoDB Atlas, and sends back `BUZZER:1` if an alert occurs!

To send data directly to the deployed cloud server:
```bash
python3 MultiSense/hardware/serial_bridge.py --cloud
```

---

### Method 2: Arduino Uno with WiFi (ESP8266 / ESP32 / WiFi Shield)
If your board has WiFi (e.g. ESP8266 ESP-01 module or ESP32):

1. Open [`arduino_esp_wifi.ino`](file:///Users/shreyashgautam/Desktop/darshana/authism/MultiSense/hardware/arduino_esp_wifi/arduino_esp_wifi.ino).
2. Enter your WiFi SSID and Password:
```cpp
const char* ssid = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";
```
3. Set your server URL:
```cpp
// Local server
const char* serverUrl = "http://192.168.1.X:8000/api/arduino/stream?format=text";
// Cloud server
const char* serverUrl = "https://multisense-ai-pslv.onrender.com/api/arduino/stream?format=text";
```
4. Flash the code. The board will transmit readings directly to the API over WiFi every 2 seconds!
