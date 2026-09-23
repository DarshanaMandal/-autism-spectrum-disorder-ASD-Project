"""
MultiSense: Hardware Sensor Stream Seeder & Telemetry Generator
Ingests realistic dummy hardware sensor streams (IR, 3-Axis ACC, Ambient Light)
through the ML prediction engine and persists records into MongoDB (or local buffer).
"""

import sys
import time
from pathlib import Path

# Add API to path
API_DIR = Path(__file__).resolve().parent
sys.path.append(str(API_DIR))

import urllib.request
import json

SAMPLE_SENSOR_READINGS = [
    {
        "description": "User Initial Baseline (Zero Sensors & Negative Lux Offset)",
        "raw_text": "IR: 0\nACC: 0, 0, 0\nLight: -2.00 lux",
        "device_id": "pediatric_node_01"
    },
    {
        "description": "Calm Baseline / Resting Standby",
        "raw_text": "IR: 72.0\nACC: 0.05, 0.02, 0.08\nLight: 120.00 lux",
        "device_id": "pediatric_node_01"
    },
    {
        "description": "Mild Motor Agitation / Arm Flutter",
        "raw_text": "IR: 118.0\nACC: 1.45, 2.10, 0.90\nLight: 380.00 lux",
        "device_id": "pediatric_node_01"
    },
    {
        "description": "Intense Bilateral Arm Flapping",
        "raw_text": "IR: 135.0\nACC: 2.80, 3.40, 1.65\nLight: 450.00 lux",
        "device_id": "pediatric_node_01"
    },
    {
        "description": "Rhythmic Head Pitching (Head Banging)",
        "raw_text": "IR: 146.0\nACC: 0.60, 4.75, 0.95\nLight: 620.00 lux",
        "device_id": "pediatric_node_01"
    },
    {
        "description": "Severe Cranial Agitation / Pain Event",
        "raw_text": "IR: 158.0\nACC: 0.85, 5.20, 1.40\nLight: 780.00 lux",
        "device_id": "pediatric_node_01"
    },
    {
        "description": "Axial Body Rotation (Spinning)",
        "raw_text": "IR: 112.0\nACC: 1.10, 0.90, 3.60\nLight: 280.00 lux",
        "device_id": "pediatric_node_01"
    },
    {
        "description": "Continuous Vestibular Seeking (Fast Spinning)",
        "raw_text": "IR: 124.0\nACC: 1.35, 1.15, 4.25\nLight: 310.00 lux",
        "device_id": "pediatric_node_01"
    },
    {
        "description": "Post-Stimming Calming / Sensory Decompression",
        "raw_text": "IR: 82.0\nACC: 0.15, 0.10, 0.25\nLight: 85.00 lux",
        "device_id": "pediatric_node_01"
    },
    {
        "description": "Deep Rest in Low Light (Dark Room)",
        "raw_text": "IR: 65.0\nACC: 0.02, 0.01, 0.03\nLight: 0.00 lux",
        "device_id": "pediatric_node_01"
    }
]


def seed_sensor_telemetry():
    api_url = "http://127.0.0.1:8000/api/sensor-stream"
    print("=" * 70)
    print("  SEEDING SENSOR TELEMETRY & RUNNING ML CLASSIFICATION")
    print("=" * 70)

    for i, sample in enumerate(SAMPLE_SENSOR_READINGS, 1):
        print(f"\n[{i}/{len(SAMPLE_SENSOR_READINGS)}] Sending: {sample['description']}")
        print("    Raw Stream Input:")
        for line in sample['raw_text'].split('\n'):
            print(f"      {line}")

        req = urllib.request.Request(
            api_url,
            data=json.dumps({
                "raw_text": sample["raw_text"],
                "device_id": sample["device_id"]
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                ml = data.get("ml_output", {})
                alert = data.get("clinical_alert", {})
                status = data.get("storage_status")

                print(f"    -> ML Prediction: {ml.get('predicted_class')} (Confidence: {ml.get('confidence', 0)*100:.1f}%)")
                print(f"    -> Clinical Alert: [{alert.get('risk_level')}] {alert.get('category')}")
                print(f"    -> Storage Status: {status} (ID: {data.get('record_id')[:8]}...)")
        except Exception as e:
            print(f"    [!] Error sending sample {i}: {e}")

        time.sleep(0.3)

    print("\n" + "=" * 70)
    print("  SEEDING COMPLETE! Checking /api/db-status...")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/api/db-status", timeout=5) as resp:
            db_status = json.loads(resp.read().decode("utf-8"))
            print(f"  - Database:          {db_status.get('database')}")
            print(f"  - Collection:        {db_status.get('collection')}")
            print(f"  - Total Stored:      {db_status.get('total_records')}")
            print(f"  - Connection Status: {db_status.get('status_message')}")
    except Exception as e:
        print("  Error fetching db status:", e)
    print("=" * 70)


if __name__ == "__main__":
    seed_sensor_telemetry()
