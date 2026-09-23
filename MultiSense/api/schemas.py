"""
MultiSense API: Pydantic Schemas for Request and Response Validation
"""

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Current server status")
    version: str = Field(default="1.0.0", description="API Version")
    device: str = Field(default="cpu", description="Computing device")
    loaded_models: List[str] = Field(description="Names of loaded machine learning models")
    supported_classes: List[str] = Field(description="List of target ASD stimming classes")


class ModelInfo(BaseModel):
    name: str
    type: str
    features_required: int
    test_accuracy: str
    macro_f1: str
    description: str


class ModelsRegistryResponse(BaseModel):
    models: Dict[str, ModelInfo]
    default_model: str


class KinematicFeaturesInput(BaseModel):
    wrist_velocity_mean: float = Field(0.45, description="Mean Euclidean velocity of wrists (m/s proxy)", ge=0.0)
    wrist_velocity_max: float = Field(1.15, description="Peak Euclidean velocity of wrists", ge=0.0)
    wrist_velocity_std: float = Field(0.20, description="Standard deviation of wrist velocity", ge=0.0)
    wrist_acceleration_mean: float = Field(0.25, description="Mean acceleration of wrists", ge=0.0)
    arm_velocity_mean: float = Field(0.38, description="Mean arm segment velocity", ge=0.0)
    head_velocity_mean: float = Field(0.14, description="Mean head velocity", ge=0.0)
    head_velocity_max: float = Field(0.30, description="Max head velocity", ge=0.0)
    head_pitch_variance: float = Field(0.04, description="Variance of vertical head pitch", ge=0.0)
    shoulder_velocity_mean: float = Field(0.12, description="Mean shoulder movement velocity", ge=0.0)
    body_velocity_mean: float = Field(0.15, description="Mean torso/body velocity", ge=0.0)
    angular_velocity_centroid: float = Field(0.08, description="Angular velocity around body centroid (rad/s)", ge=0.0)
    movement_intensity: float = Field(0.35, description="Combined kinematic kinetic energy proxy", ge=0.0)
    pose_variance: float = Field(0.03, description="Overall skeletal landmark spatial variance", ge=0.0)
    elbow_angle_mean: float = Field(95.0, description="Mean elbow flexion angle in degrees", ge=0.0, le=180.0)
    elbow_angle_std: float = Field(15.0, description="Standard deviation of elbow angle", ge=0.0)
    shoulder_angle_mean: float = Field(65.0, description="Mean shoulder abduction angle in degrees", ge=0.0, le=180.0)
    shoulder_angle_std: float = Field(12.0, description="Standard deviation of shoulder angle", ge=0.0)
    model_choice: Optional[str] = Field("gan_augmented", description="Model to use: 'gan_augmented' or 'baseline'")

    class Config:
        json_schema_extra = {
            "example": {
                "wrist_velocity_mean": 0.48,
                "wrist_velocity_max": 1.25,
                "wrist_velocity_std": 0.22,
                "wrist_acceleration_mean": 0.28,
                "arm_velocity_mean": 0.40,
                "head_velocity_mean": 0.15,
                "head_velocity_max": 0.32,
                "head_pitch_variance": 0.04,
                "shoulder_velocity_mean": 0.14,
                "body_velocity_mean": 0.16,
                "angular_velocity_centroid": 0.09,
                "movement_intensity": 0.42,
                "pose_variance": 0.035,
                "elbow_angle_mean": 105.0,
                "elbow_angle_std": 18.0,
                "shoulder_angle_mean": 70.0,
                "shoulder_angle_std": 14.0,
                "model_choice": "gan_augmented"
            }
        }


class MultimodalInput(KinematicFeaturesInput):
    max30102_hr_bpm: float = Field(118.0, description="Heart Rate in Beats Per Minute (MAX30102 PPG)", ge=40.0, le=220.0)
    max30102_spo2_pct: float = Field(97.5, description="Blood Oxygen Saturation % (MAX30102 PPG)", ge=80.0, le=100.0)
    mpu6050_acc_mag_mean: float = Field(3.1, description="Mean 3D linear acceleration magnitude in g (MPU6050 IMU)", ge=0.0)
    mpu6050_gyro_yaw_rads: float = Field(0.55, description="Mean yaw rotational rate in rad/s (MPU6050 IMU)", ge=0.0)
    bh1750_lux_mean: float = Field(480.0, description="Ambient illuminance in Lux (BH1750 Light Sensor)", ge=0.0)

    class Config:
        json_schema_extra = {
            "example": {
                "wrist_velocity_mean": 0.18,
                "wrist_velocity_max": 0.45,
                "wrist_velocity_std": 0.08,
                "wrist_acceleration_mean": 0.12,
                "arm_velocity_mean": 0.16,
                "head_velocity_mean": 0.52,
                "head_velocity_max": 1.45,
                "head_pitch_variance": 0.38,
                "shoulder_velocity_mean": 0.31,
                "body_velocity_mean": 0.28,
                "angular_velocity_centroid": 0.11,
                "movement_intensity": 0.65,
                "pose_variance": 0.052,
                "elbow_angle_mean": 85.0,
                "elbow_angle_std": 12.0,
                "shoulder_angle_mean": 55.0,
                "shoulder_angle_std": 10.0,
                "max30102_hr_bpm": 142.0,
                "max30102_spo2_pct": 96.2,
                "mpu6050_acc_mag_mean": 4.9,
                "mpu6050_gyro_yaw_rads": 0.42,
                "bh1750_lux_mean": 650.0
            }
        }


class PoseSequenceInput(BaseModel):
    # Accepts array of shape (T, 132) or (T, 33, 4)
    sequence: List[List[float]] = Field(
        ...,
        description="Temporal sequence of 33 body landmarks [x, y, z, v] flattened (132 values per frame, 30-120 frames recommended)"
    )
    model_choice: Optional[str] = Field("gan_augmented", description="Model choice: 'gan_augmented' or 'baseline'")


class ClinicalAlert(BaseModel):
    risk_level: str = Field(description="Alert level: 'HIGH', 'MEDIUM', or 'LOW'")
    risk_color: str = Field(description="Color code: '#ef4444' (red), '#f59e0b' (amber), '#10b981' (green)")
    category: str = Field(description="Category of stimming behavior")
    clinical_note: str = Field(description="Medical and sensory explanation of behavior")
    caregiver_guidance: str = Field(description="Actionable guidance for parents/caregivers/clinicians")


class PredictionResponse(BaseModel):
    success: bool = True
    predicted_class: str = Field(description="Predicted stimming behavior (ArmFlapping, HeadBanging, Spinning)")
    class_id: int = Field(description="Numerical class index (0, 1, 2)")
    confidence: float = Field(description="Confidence probability (0.0 - 1.0)")
    probabilities: Dict[str, float] = Field(description="Per-class predicted probability distribution")
    model_used: str = Field(description="Identifier of model executing prediction")
    clinical_alert: ClinicalAlert = Field(description="Clinical risk rating and actionable guidance")
    biomechanical_summary: Optional[Dict[str, float]] = Field(None, description="Top discriminative kinematic metrics")


class SyntheticGenerationRequest(BaseModel):
    behavior_class: str = Field(
        default="ArmFlapping",
        description="Target class: 'ArmFlapping', 'HeadBanging', or 'Spinning'"
    )
    num_samples: int = Field(default=5, ge=1, le=50, description="Number of sequences to generate")
    noise_seed: Optional[int] = Field(None, description="Optional random seed for reproducibility")


class SyntheticGenerationResponse(BaseModel):
    success: bool = True
    behavior_class: str
    num_samples: int
    sequence_shape: List[int]
    sample_preview: List[List[float]]
    kinematic_stats: Dict[str, float]
    info: str


class RawSensorInput(BaseModel):
    raw_text: Optional[str] = Field(
        None,
        description="Raw multiline string directly from microcontroller, e.g. 'IR: 0\\nACC: 0, 0, 0\\nLight: -2.00 lux'"
    )
    ir: Optional[float] = Field(
        None,
        description="Infrared / PPG sensor value (e.g. 0 when idle/no skin contact, >0 when heart pulse detected)"
    )
    acc_x: Optional[float] = Field(None, description="MPU6050 accelerometer X axis")
    acc_y: Optional[float] = Field(None, description="MPU6050 accelerometer Y axis")
    acc_z: Optional[float] = Field(None, description="MPU6050 accelerometer Z axis")
    light_lux: Optional[float] = Field(
        None,
        description="BH1750 ambient light lux reading (e.g. -2.00 lux uncalibrated offset or 350.0 lux)"
    )
    device_id: Optional[str] = Field("sensor_node_01", description="Identifier of the physical device or subject")

    class Config:
        json_schema_extra = {
            "example": {
                "raw_text": "IR: 0\nACC: 0, 0, 0\nLight: -2.00 lux",
                "device_id": "child_wearable_node_01"
            }
        }


class SensorRecordResponse(BaseModel):
    success: bool = True
    record_id: str
    timestamp: str
    device_id: str
    storage_status: str = Field(description="'SAVED_TO_MONGODB' or 'BUFFERED_LOCALLY'")
    input_sensors: Dict[str, Any]
    computed_features: Dict[str, Any]
    ml_output: Dict[str, Any]
    clinical_alert: ClinicalAlert


class SensorHistoryResponse(BaseModel):
    success: bool = True
    total_returned: int
    storage_status: str
    records: List[Dict[str, Any]]


class DatabaseStatusResponse(BaseModel):
    has_uri: bool
    connected: bool
    masked_uri: str
    database: str
    collection: str
    total_records: int
    database_records: int
    buffered_in_memory: int
    status_message: str


class ArduinoCompactResponse(BaseModel):
    ok: bool = True
    pred: str = Field(description="Predicted stimming class: ArmFlapping, HeadBanging, Spinning")
    conf: float = Field(description="Model confidence probability (0.0 to 1.0)")
    risk: str = Field(description="Clinical risk level: HIGH, MEDIUM, or LOW")
    buzzer: int = Field(description="1 if alert requires physical notification (LED/Buzzer), 0 otherwise")
    db: str = Field(description="'SAVED' if persisted to MongoDB, 'BUFFERED' if local")
    rec_id: str = Field(description="Unique record identifier")
