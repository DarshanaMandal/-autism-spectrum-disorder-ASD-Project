"""
Test script for MultiSense FastAPI endpoints using TestClient
"""

import sys
from pathlib import Path
import json

from fastapi.testclient import TestClient

# Setup sys.path
API_DIR = Path(__file__).resolve().parent
sys.path.append(str(API_DIR))
sys.path.append(str(API_DIR.parent))

from main import app

client = TestClient(app)

def run_api_tests():
    print("\n=======================================================")
    print("  RUNNING COMPREHENSIVE MULTISENSE API TEST SUITE")
    print("=======================================================\n")

    # 1. Test Health
    print("[*] Testing GET /health ...")
    res = client.get("/health")
    assert res.status_code == 200, f"Health failed: {res.text}"
    health_data = res.json()
    print(f"    [✓] Status: {health_data['status']}, Loaded Models: {health_data['loaded_models']}")

    # 2. Test Models Registry
    print("\n[*] Testing GET /models ...")
    res = client.get("/models")
    assert res.status_code == 200, f"Models failed: {res.text}"
    models_data = res.json()
    print(f"    [✓] Available models: {list(models_data['models'].keys())}")

    # 3. Test Feature Prediction (ArmFlapping Simulation)
    print("\n[*] Testing POST /predict/features (Simulating ArmFlapping) ...")
    payload = {
        "wrist_velocity_mean": 0.52,
        "wrist_velocity_max": 1.35,
        "wrist_velocity_std": 0.22,
        "wrist_acceleration_mean": 0.28,
        "arm_velocity_mean": 0.40,
        "head_velocity_mean": 0.12,
        "head_velocity_max": 0.28,
        "head_pitch_variance": 0.03,
        "shoulder_velocity_mean": 0.14,
        "body_velocity_mean": 0.15,
        "angular_velocity_centroid": 0.08,
        "movement_intensity": 0.45,
        "pose_variance": 0.035,
        "elbow_angle_mean": 105.0,
        "elbow_angle_std": 18.0,
        "shoulder_angle_mean": 70.0,
        "shoulder_angle_std": 14.0,
        "model_choice": "gan_augmented"
    }
    res = client.post("/predict/features", json=payload)
    assert res.status_code == 200, f"Predict features failed: {res.text}"
    pred_data = res.json()
    print(f"    [✓] Predicted Class: {pred_data['predicted_class']}")
    print(f"    [✓] Confidence: {pred_data['confidence'] * 100:.2f}%")
    print(f"    [✓] Probabilities: {pred_data['probabilities']}")
    print(f"    [✓] Clinical Alert Risk: {pred_data['clinical_alert']['risk_level']} - {pred_data['clinical_alert']['category']}")

    # 4. Test Multimodal Sensor Fusion (Simulating HeadBanging + High HR)
    print("\n[*] Testing POST /predict/multimodal (HeadBanging + Tachycardia) ...")
    multi_payload = dict(payload)
    multi_payload.update({
        "head_velocity_mean": 0.58,
        "head_pitch_variance": 0.42,
        "angular_velocity_centroid": 0.10,
        "max30102_hr_bpm": 145.0,
        "max30102_spo2_pct": 96.0,
        "mpu6050_acc_mag_mean": 4.8,
        "mpu6050_gyro_yaw_rads": 0.38,
        "bh1750_lux_mean": 650.0
    })
    res = client.post("/predict/multimodal", json=multi_payload)
    assert res.status_code == 200, f"Multimodal failed: {res.text}"
    multi_data = res.json()
    print(f"    [✓] Predicted Class: {multi_data['predicted_class']}")
    print(f"    [✓] Confidence: {multi_data['confidence'] * 100:.2f}%")
    print(f"    [✓] Alert Level: {multi_data['clinical_alert']['risk_level']}")
    print(f"    [✓] Caregiver Guidance: {multi_data['clinical_alert']['caregiver_guidance']}")

    # 5. Test GAN Synthetic Generation
    print("\n[*] Testing POST /generate/synthetic ...")
    gan_payload = {
        "behavior_class": "Spinning",
        "num_samples": 3,
        "noise_seed": 42
    }
    res = client.post("/generate/synthetic", json=gan_payload)
    assert res.status_code == 200, f"GAN failed: {res.text}"
    gan_data = res.json()
    print(f"    [✓] Generated shape: {gan_data['sequence_shape']}")
    print(f"    [✓] Info: {gan_data['info']}")

    # 6. Test Video Upload Inference
    sample_video = API_DIR.parent / "data" / "processed" / "ArmFlapping" / "v_ArmFlapping_01_b_01.mp4"
    if sample_video.exists():
        print(f"\n[*] Testing POST /predict/video with {sample_video.name} ...")
        with open(sample_video, "rb") as f:
            res = client.post("/predict/video", files={"file": ("test.mp4", f, "video/mp4")})
        assert res.status_code == 200, f"Video prediction failed: {res.text}"
        video_pred = res.json()
        print(f"    [✓] Video Predicted: {video_pred['predicted_class']} ({video_pred['confidence']*100:.1f}%)")
        print(f"    [✓] Probabilities: {video_pred['probabilities']}")
        print(f"    [✓] Alert Level: {video_pred['clinical_alert']['risk_level']}")
    else:
        print("\n[!] Sample video clip not found, skipping video test.")

    print("\n=======================================================")
    print("  ALL API ENDPOINTS TESTED SUCCESSFULLY! [PASS]")
    print("=======================================================\n")

if __name__ == "__main__":
    run_api_tests()
