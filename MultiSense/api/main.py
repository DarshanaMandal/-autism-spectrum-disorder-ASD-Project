"""
MultiSense: Production-Grade FastAPI Endpoints for ASD Stimming Detection
Provides high-performance, async REST endpoints for:
- Kinematic Feature Classification
- Multimodal Wearable Sensor Fusion
- Direct Skeletal Pose Keypoint Processing
- Full Video Upload Inference (MediaPipe Pose -> Feature Extraction -> Random Forest)
- Conditional Bi-GRU GAN Synthetic Generation
- Interactive Dashboard (/dashboard)
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

# Setup local imports
API_DIR = Path(__file__).resolve().parent
BASE_DIR = API_DIR.parent
FRONTEND_DIR = BASE_DIR / "frontend"
sys.path.append(str(BASE_DIR))
sys.path.append(str(API_DIR))

from schemas import (
    HealthResponse,
    ModelsRegistryResponse,
    ModelInfo,
    KinematicFeaturesInput,
    MultimodalInput,
    PoseSequenceInput,
    PredictionResponse,
    SyntheticGenerationRequest,
    SyntheticGenerationResponse,
    RawSensorInput,
    SensorRecordResponse,
    SensorHistoryResponse,
    DatabaseStatusResponse,
    ArduinoCompactResponse
)
import inference
from database import db_manager
from sensor_parser import process_sensor_input

# Initialize FastAPI App
app = FastAPI(
    title="MultiSense ASD Stimming Detection API",
    description=(
        "Research & Clinical Grade REST API for Autism Spectrum Disorder (ASD) "
        "Self-Stimulatory Behavior Detection (ArmFlapping, HeadBanging, Spinning). "
        "Integrates Biomechanical Pose Kinematics, Conditional Recurrent GAN Augmentation, "
        "and Wearable Multimodal Sensor Fusion (MAX30102 PPG + MPU6050 IMU + BH1750 Lux)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin frontend apps (Vercel, Netlify, localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories for modern clinical frontend
if (FRONTEND_DIR / "css").exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
if (FRONTEND_DIR / "js").exists():
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
if (FRONTEND_DIR / "samples").exists():
    app.mount("/samples", StaticFiles(directory=str(FRONTEND_DIR / "samples")), name="samples")


@app.get("/", tags=["User Interface"])
async def root_dashboard():
    """Serves the MultiSense interactive clinical web dashboard."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Frontend index.html not found"}


@app.get("/dashboard", tags=["User Interface"])
async def dashboard_redirect():
    """Alternative route for the interactive clinical dashboard."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Frontend index.html not found"}


@app.get("/api/info", tags=["System Information"])
async def api_info():
    """API metadata, loaded models, and target classes."""
    return {
        "title": "MultiSense ASD Stimming Detection & Biomechanical Pose API",
        "status": "online",
        "version": "1.0.0",
        "documentation": "/docs",
        "interactive_dashboard": "/",
        "models_available": [
            "Baseline RF (84.65% Accuracy)",
            "GAN-Augmented RF (87.98% Accuracy, +5.67% ArmFlapping Recall)",
            "Multimodal Fusion RF (97.44% Accuracy, Video + PPG + IMU + Lux)"
        ],
        "target_behaviors": ["ArmFlapping", "HeadBanging", "Spinning"]
    }


@app.get("/health", response_model=HealthResponse, tags=["System Information"])
async def health_check():
    """Service health verification endpoint."""
    loaded = []
    if inference.manager.baseline_rf is not None:
        loaded.append("Baseline RF")
    if inference.manager.gan_rf is not None:
        loaded.append("GAN-Augmented RF")
    if inference.manager.multimodal_rf is not None:
        loaded.append("Multimodal Sensor Fusion RF")
    if inference.manager.gan_generator is not None:
        loaded.append("Conditional Bi-GRU GAN Generator")

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=str(inference.manager.device),
        loaded_models=loaded,
        supported_classes=inference.CLASS_NAMES
    )


@app.get("/models", response_model=ModelsRegistryResponse, tags=["Models Registry"])
async def get_models_registry():
    """Detailed metadata and scientific metrics for all trained models."""
    registry = {
        "gan_augmented": ModelInfo(
            name="GAN-Augmented Random Forest (Recommended)",
            type="Balanced Ensemble Classifier (300 Trees)",
            features_required=17,
            test_accuracy="87.98%",
            macro_f1="0.8307",
            description="Trained on real SSBD sequences + 1,200 synthetic pose trajectories to eliminate minority-class false negatives."
        ),
        "baseline": ModelInfo(
            name="Baseline Random Forest (No GAN)",
            type="Balanced Ensemble Classifier (300 Trees)",
            features_required=17,
            test_accuracy="84.65%",
            macro_f1="0.7883",
            description="Trained strictly on real SSBD video data. Shows natural difficulty on minority class ArmFlapping."
        ),
        "multimodal": ModelInfo(
            name="Multimodal Sensor Fusion Random Forest",
            type="Fused Kinematic + Biometric Ensemble Classifier",
            features_required=22,
            test_accuracy="97.44%",
            macro_f1="0.9596",
            description="Fuses 17 kinematic features with MAX30102 Heart Rate/SpO2, MPU6050 6-DOF IMU, and BH1750 Ambient Lux."
        )
    }
    return ModelsRegistryResponse(models=registry, default_model="gan_augmented")


@app.post("/predict/features", response_model=PredictionResponse, tags=["Inference"])
async def predict_features(payload: KinematicFeaturesInput):
    """
    Predicts ASD stimming behavior from 17 biomechanical kinematic features.
    Allows selecting between 'gan_augmented' and 'baseline' models.
    """
    try:
        data = payload.dict()
        choice = data.pop("model_choice", "gan_augmented")
        result = inference.predict_from_features(data, model_choice=choice)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/multimodal", response_model=PredictionResponse, tags=["Inference"])
async def predict_multimodal(payload: MultimodalInput):
    """
    Predicts ASD stimming behavior using the 97.44% Multimodal Sensor Fusion model.
    Accepts 17 kinematic pose features + 5 wearable biometric features (MAX30102 + MPU6050 + BH1750).
    """
    try:
        data = payload.dict()
        data.pop("model_choice", None)
        result = inference.predict_from_multimodal(data)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/pose", response_model=PredictionResponse, tags=["Inference"])
async def predict_from_pose(payload: PoseSequenceInput):
    """
    Predicts ASD stimming directly from raw/normalized skeletal pose landmark sequences.
    Accepts (T, 132) array representing (x, y, z, visibility) for 33 landmarks over time.
    """
    try:
        arr = np.array(payload.sequence, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 132:
            raise HTTPException(
                status_code=400,
                detail=f"Expected array of shape (T, 132), got {arr.shape}"
            )
        # Extract features
        feats = inference.extract_features_from_pose_sequence(arr)
        result = inference.predict_from_features(feats, model_choice=payload.model_choice or "gan_augmented")
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/video", response_model=PredictionResponse, tags=["Inference"])
async def predict_video(
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV) containing child movement"),
    model_choice: str = Query("gan_augmented", description="Model choice: 'gan_augmented' or 'baseline'")
):
    """
    Upload a video file to perform end-to-end classification:
    1. MediaPipe 33-point skeletal landmark extraction across all frames
    2. Kinematic feature engineering (velocities, head pitch variance, rotational speeds)
    3. Classification and structured Clinical Risk Alert generation.
    """
    # Validate extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(status_code=400, detail="Only video files (.mp4, .avi, .mov, .mkv) are supported.")

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Extract features directly from video kinematics
        feats = inference.extract_features_from_video(tmp_path)
        # Predict
        result = inference.predict_from_features(feats, model_choice=model_choice)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/generate/synthetic", response_model=SyntheticGenerationResponse, tags=["GAN Generator"])
async def generate_synthetic(payload: SyntheticGenerationRequest):
    """
    Generates synthetic 60-frame pose sequences using the Conditional Bi-GRU Recurrent GAN.
    """
    try:
        result = inference.generate_synthetic_samples(
            behavior_class=payload.behavior_class,
            num_samples=payload.num_samples,
            seed=payload.noise_seed
        )
        return SyntheticGenerationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Hardware Sensor Stream & MongoDB Telemetry Endpoints
# ==============================================================================

@app.post("/api/sensor-stream", response_model=SensorRecordResponse, tags=["Sensor Telemetry & Database"])
@app.post("/api/sensor/predict-and-store", response_model=SensorRecordResponse, tags=["Sensor Telemetry & Database"])
async def ingest_sensor_stream(payload: RawSensorInput):
    """
    Ingests live sensor readings from hardware (IR, 3-Axis ACC, Ambient Light).
    Supports raw microcontroller stream text (e.g. 'IR: 0\\nACC: 0, 0, 0\\nLight: -2.00 lux')
    or structured JSON. Fuses inputs into the 22-feature Multimodal RF classifier,
    computes clinical risk guidance, and appends the complete record to MongoDB.
    """
    try:
        raw_summary, computed_telemetry, multimodal_feats = process_sensor_input(payload)
        # Execute 97.44% Multimodal Classifier
        ml_res = inference.predict_from_multimodal(multimodal_feats)

        # Store in MongoDB (or local buffer if DB not yet connected)
        record = db_manager.save_record(
            raw_input=raw_summary,
            computed_features=computed_telemetry,
            ml_prediction=ml_res,
            clinical_alert=ml_res["clinical_alert"],
            device_id=payload.device_id or "sensor_node_01"
        )

        return SensorRecordResponse(
            success=True,
            record_id=record["_id"],
            timestamp=record["timestamp"],
            device_id=record["device_id"],
            storage_status=record["storage_status"],
            input_sensors=record["input_sensors"],
            computed_features=record["computed_features"],
            ml_output=record["ml_output"],
            clinical_alert=record["clinical_alert"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sensor stream ingestion failed: {str(e)}")


# ==============================================================================
# Dedicated Arduino Uno & Microcontroller Endpoints
# ==============================================================================

@app.post("/api/arduino/stream", tags=["Arduino & Microcontrollers"])
@app.get("/api/arduino/stream", tags=["Arduino & Microcontrollers"])
async def arduino_sensor_stream(
    request: Request,
    format: str = Query("json", description="Response format: 'json' or 'text'"),
    ir: Optional[float] = Query(None, description="Infrared reading (e.g. 0 or pulse value)"),
    ax: Optional[float] = Query(None, description="MPU6050 Accelerometer X"),
    ay: Optional[float] = Query(None, description="MPU6050 Accelerometer Y"),
    az: Optional[float] = Query(None, description="MPU6050 Accelerometer Z"),
    lux: Optional[float] = Query(None, description="BH1750 Ambient Light lux"),
    device: Optional[str] = Query("arduino_uno", description="Microcontroller Device ID")
):
    """
    Ultra-lightweight endpoint specifically designed for microcontrollers (Arduino Uno, ESP8266, ESP32):
    - Supports HTTP GET queries: /api/arduino/stream?ir=0&ax=0&ay=0&az=0&lux=-2.0
    - Supports HTTP POST with raw microcontroller text ('IR: 0\\nACC: 0, 0, 0\\nLight: -2.00 lux')
    - Supports HTTP POST with JSON body
    - Returns compact responses (low memory footprint to prevent Arduino 2KB RAM overflow):
        * format=json: {"ok":true,"pred":"ArmFlapping","conf":0.45,"risk":"MEDIUM","buzzer":0,"db":"SAVED"}
        * format=text: PRED:ArmFlapping|CONF:0.45|RISK:MEDIUM|BUZZER:0|DB:SAVED
    """
    raw_text = None
    input_ir = ir
    input_ax = ax
    input_ay = ay
    input_az = az
    input_lux = lux
    device_id = device or "arduino_uno"

    # If POST, check body content
    if request.method == "POST":
        try:
            body_bytes = await request.body()
            body_str = body_bytes.decode("utf-8", errors="ignore").strip()
            if body_str.startswith("{"):
                # JSON body
                import json
                parsed_json = json.loads(body_str)
                raw_text = parsed_json.get("raw_text")
                if input_ir is None:
                    input_ir = parsed_json.get("ir")
                if input_ax is None:
                    input_ax = parsed_json.get("acc_x", parsed_json.get("ax"))
                if input_ay is None:
                    input_ay = parsed_json.get("acc_y", parsed_json.get("ay"))
                if input_az is None:
                    input_az = parsed_json.get("acc_z", parsed_json.get("az"))
                if input_lux is None:
                    input_lux = parsed_json.get("light_lux", parsed_json.get("lux"))
                if parsed_json.get("device_id"):
                    device_id = parsed_json.get("device_id")
            elif body_str:
                # Raw text format (e.g. IR: 0\nACC: 0, 0, 0\nLight: -2.00 lux)
                raw_text = body_str
        except Exception:
            pass

    # Build RawSensorInput
    payload = RawSensorInput(
        raw_text=raw_text,
        ir=input_ir,
        acc_x=input_ax,
        acc_y=input_ay,
        acc_z=input_az,
        light_lux=input_lux,
        device_id=device_id
    )

    try:
        raw_summary, computed_telemetry, multimodal_feats = process_sensor_input(payload)
        # Run ML inference
        ml_res = inference.predict_from_multimodal(multimodal_feats)

        # Persist in MongoDB
        record = db_manager.save_record(
            raw_input=raw_summary,
            computed_features=computed_telemetry,
            ml_prediction=ml_res,
            clinical_alert=ml_res["clinical_alert"],
            device_id=device_id
        )

        pred_class = ml_res["predicted_class"]
        confidence = round(float(ml_res["confidence"]), 4)
        risk_level = ml_res["clinical_alert"]["risk_level"]
        # Buzzer flag: 1 if HIGH risk (SIB / HeadBanging), 0 otherwise
        buzzer = 1 if risk_level == "HIGH" else 0
        db_stat = "SAVED" if record["storage_status"] == "SAVED_TO_MONGODB" else "BUFFERED"
        rec_id = record["_id"][:8]

        # Plain text output for simple Arduino parsing without JSON libraries
        if format.lower() == "text":
            text_resp = f"PRED:{pred_class}|CONF:{confidence:.2f}|RISK:{risk_level}|BUZZER:{buzzer}|DB:{db_stat}"
            return PlainTextResponse(text_resp)

        # Compact JSON output
        return ArduinoCompactResponse(
            ok=True,
            pred=pred_class,
            conf=confidence,
            risk=risk_level,
            buzzer=buzzer,
            db=db_stat,
            rec_id=rec_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Arduino stream processing error: {str(e)}")


@app.get("/api/sensor-history", response_model=SensorHistoryResponse, tags=["Sensor Telemetry & Database"])
async def get_sensor_history(limit: int = Query(50, ge=1, le=500, description="Max records to retrieve")):
    """
    Fetches recent sensor telemetry and classification records from MongoDB collection 'sensor_telemetry'.
    """
    try:
        records = db_manager.get_history(limit=limit)
        status_info = db_manager.get_status()
        storage = "MONGODB" if status_info["connected"] else "IN_MEMORY_BUFFER"
        return SensorHistoryResponse(
            success=True,
            total_returned=len(records),
            storage_status=storage,
            records=records
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch sensor history: {str(e)}")


@app.get("/api/db-status", response_model=DatabaseStatusResponse, tags=["Sensor Telemetry & Database"])
async def get_database_status():
    """
    Returns current connection status to MongoDB Atlas / local database.
    """
    status_info = db_manager.get_status()
    return DatabaseStatusResponse(**status_info)


@app.post("/api/db-configure", tags=["Sensor Telemetry & Database"])
async def configure_database(payload: Dict[str, Any]):
    """
    Dynamically update or test MONGODB_URI at runtime without restarting server.
    """
    uri = payload.get("mongodb_uri", "").strip()
    db_name = payload.get("db_name", "").strip()
    coll_name = payload.get("collection", "").strip()

    if uri:
        os.environ["MONGODB_URI"] = uri
    if db_name:
        os.environ["MONGODB_DB_NAME"] = db_name
    if coll_name:
        os.environ["MONGODB_COLLECTION"] = coll_name

    # Try connecting
    connected, msg = db_manager.connect()
    status_info = db_manager.get_status()
    return {
        "success": connected,
        "connected": connected,
        "message": msg,
        "status": status_info
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("  STARTING MULTISENSE ASD STIMMING API SERVER")
    print("  Endpoints:")
    print("    - Web Portal:          http://127.0.0.1:8000/")
    print("    - REST API Docs:       http://127.0.0.1:8000/docs")
    print("="*70 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
