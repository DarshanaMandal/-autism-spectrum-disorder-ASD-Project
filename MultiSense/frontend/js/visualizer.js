/**
 * MultiSense Visualizer Engine
 * Real-time Canvas Rendering for:
 * 1. Cardiac ECG Telemetry Waveform (MAX30102 PPG)
 * 2. 2D/3D Biomechanical Skeletal Pose Animator (TimeGAN)
 * 3. Video Motion Particle/Landmark Overlay
 */

class VisualizerEngine {
    constructor() {
        this.ecgCanvas = null;
        this.ecgCtx = null;
        this.ecgX = 0;
        this.currentBpm = 80;
        this.ecgAnimationId = null;

        this.poseCanvas = null;
        this.poseCtx = null;
        this.currentPoseFrames = null;
        this.currentPoseFrameIdx = 0;
        this.poseAnimationId = null;
        this.currentClass = "ArmFlapping";

        this.initECG();
    }

    /* =========================================================
       1. REAL-TIME CARDIAC ECG MONITOR (MAX30102)
       ========================================================= */
    initECG() {
        this.ecgCanvas = document.getElementById("ecgCanvas");
        if (!this.ecgCanvas) return;
        this.ecgCtx = this.ecgCanvas.getContext("2d");
        this.resizeECG();
        window.addEventListener("resize", () => this.resizeECG());
        this.startECGLoop();
    }

    resizeECG() {
        if (!this.ecgCanvas) return;
        const rect = this.ecgCanvas.getBoundingClientRect();
        this.ecgCanvas.width = rect.width * window.devicePixelRatio || 300;
        this.ecgCanvas.height = rect.height * window.devicePixelRatio || 90;
        this.ecgCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }

    setBPM(bpm) {
        this.currentBpm = Math.max(40, Math.min(200, bpm));
    }

    startECGLoop() {
        let phase = 0;
        const draw = () => {
            if (!this.ecgCtx || !this.ecgCanvas) return;
            const width = this.ecgCanvas.getBoundingClientRect().width;
            const height = this.ecgCanvas.getBoundingClientRect().height;
            const midY = height / 2;

            // Fade background slightly for phosphor CRT decay effect
            this.ecgCtx.fillStyle = "rgba(4, 8, 16, 0.08)";
            this.ecgCtx.fillRect(0, 0, width, height);

            // Calculate cardiac P-Q-R-S-T wave function based on BPM
            const speed = (this.currentBpm / 60) * 0.08;
            phase = (phase + speed) % 1;

            let yOffset = 0;
            // P Wave (at phase 0.15 - 0.25)
            if (phase > 0.15 && phase < 0.25) {
                yOffset = -Math.sin((phase - 0.15) * 10 * Math.PI) * 5;
            }
            // Q Dip (at phase 0.33)
            else if (phase >= 0.32 && phase < 0.35) {
                yOffset = 4;
            }
            // R Peak (sharp spike at 0.38)
            else if (phase >= 0.35 && phase < 0.41) {
                yOffset = -Math.sin((phase - 0.35) * (Math.PI / 0.06)) * (height * 0.42);
            }
            // S Dip (at phase 0.43)
            else if (phase >= 0.41 && phase < 0.45) {
                yOffset = 8;
            }
            // T Wave (at phase 0.55 - 0.70)
            else if (phase >= 0.55 && phase < 0.70) {
                yOffset = -Math.sin((phase - 0.55) * (Math.PI / 0.15)) * 9;
            }

            const currentY = midY + yOffset;

            // Draw ECG trace
            this.ecgCtx.beginPath();
            this.ecgCtx.lineWidth = 2.2;
            const isTachycardia = this.currentBpm >= 130;
            this.ecgCtx.strokeStyle = isTachycardia ? "#ef4444" : "#10b981";
            this.ecgCtx.shadowColor = isTachycardia ? "rgba(239, 68, 68, 0.8)" : "rgba(16, 185, 129, 0.8)";
            this.ecgCtx.shadowBlur = 8;

            this.ecgCtx.moveTo(this.ecgX, this.lastY || midY);
            this.ecgX = (this.ecgX + 2.5) % width;
            this.ecgCtx.lineTo(this.ecgX, currentY);
            this.ecgCtx.stroke();

            this.lastY = currentY;
            this.ecgAnimationId = requestAnimationFrame(draw);
        };
        draw();
    }

    /* =========================================================
       2. BIOMECHANICAL SKELETAL POSE ANIMATOR (GAN / KINEMATICS)
       ========================================================= */
    initPoseAnimator() {
        this.poseCanvas = document.getElementById("poseCanvas");
        if (!this.poseCanvas) return;
        this.poseCtx = this.poseCanvas.getContext("2d");
        this.resizePose();
        window.addEventListener("resize", () => this.resizePose());
        this.loadSampleSequence("ArmFlapping");
        this.startPoseLoop();
    }

    resizePose() {
        if (!this.poseCanvas) return;
        const rect = this.poseCanvas.getBoundingClientRect();
        this.poseCanvas.width = rect.width * window.devicePixelRatio || 300;
        this.poseCanvas.height = rect.height * window.devicePixelRatio || 280;
        this.poseCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }

    loadSampleSequence(behaviorClass, syntheticSequence = null) {
        this.currentClass = behaviorClass;
        this.currentPoseFrameIdx = 0;

        if (syntheticSequence && Array.isArray(syntheticSequence)) {
            this.currentPoseFrames = syntheticSequence;
            return;
        }

        // Procedural realistic 60-frame kinematic skeletal loop generator
        const frames = [];
        const T = 60;
        for (let t = 0; t < T; t++) {
            const phase = (t / T) * Math.PI * 2;
            const joints = {};

            // Torso center
            joints.hip = { x: 0.5, y: 0.65 };
            joints.neck = { x: 0.5, y: 0.35 };

            if (behaviorClass === "ArmFlapping") {
                // Rapid wrist flutter
                const flapSpeed = 4;
                const flapAmp = Math.sin(phase * flapSpeed) * 0.16;
                joints.head = { x: 0.5, y: 0.22 + Math.sin(phase * 2) * 0.02 };
                joints.l_shoulder = { x: 0.40, y: 0.35 };
                joints.r_shoulder = { x: 0.60, y: 0.35 };
                joints.l_elbow = { x: 0.32, y: 0.44 + flapAmp * 0.5 };
                joints.r_elbow = { x: 0.68, y: 0.44 + flapAmp * 0.5 };
                joints.l_wrist = { x: 0.28, y: 0.30 + flapAmp };
                joints.r_wrist = { x: 0.72, y: 0.30 + flapAmp };
            } else if (behaviorClass === "HeadBanging") {
                // Sagittal pitch motion
                const pitchAmp = Math.sin(phase * 2) * 0.14;
                joints.head = { x: 0.5, y: 0.22 + pitchAmp };
                joints.neck = { x: 0.5, y: 0.35 + pitchAmp * 0.6 };
                joints.l_shoulder = { x: 0.41, y: 0.37 + pitchAmp * 0.3 };
                joints.r_shoulder = { x: 0.59, y: 0.37 + pitchAmp * 0.3 };
                joints.l_elbow = { x: 0.36, y: 0.50 };
                joints.r_elbow = { x: 0.64, y: 0.50 };
                joints.l_wrist = { x: 0.42, y: 0.42 + pitchAmp * 0.4 };
                joints.r_wrist = { x: 0.58, y: 0.42 + pitchAmp * 0.4 };
            } else {
                // Spinning (axial rotation projection)
                const spinCos = Math.cos(phase * 2);
                const spinSin = Math.sin(phase * 2);
                joints.head = { x: 0.5 + spinSin * 0.03, y: 0.22 };
                joints.l_shoulder = { x: 0.5 - 0.12 * spinCos, y: 0.35 };
                joints.r_shoulder = { x: 0.5 + 0.12 * spinCos, y: 0.35 };
                joints.l_elbow = { x: 0.5 - 0.18 * spinCos, y: 0.48 };
                joints.r_elbow = { x: 0.5 + 0.18 * spinCos, y: 0.48 };
                joints.l_wrist = { x: 0.5 - 0.22 * spinCos, y: 0.45 };
                joints.r_wrist = { x: 0.5 + 0.22 * spinCos, y: 0.45 };
            }

            // Lower limbs
            joints.l_hip = { x: 0.44, y: 0.65 };
            joints.r_hip = { x: 0.56, y: 0.65 };
            joints.l_knee = { x: 0.44, y: 0.78 };
            joints.r_knee = { x: 0.56, y: 0.78 };
            joints.l_ankle = { x: 0.44, y: 0.90 };
            joints.r_ankle = { x: 0.56, y: 0.90 };

            frames.push(joints);
        }
        this.currentPoseFrames = frames;
    }

    startPoseLoop() {
        const render = () => {
            if (!this.poseCtx || !this.poseCanvas || !this.currentPoseFrames) return;
            const w = this.poseCanvas.getBoundingClientRect().width;
            const h = this.poseCanvas.getBoundingClientRect().height;

            // Clear
            this.poseCtx.fillStyle = "#030712";
            this.poseCtx.fillRect(0, 0, w, h);

            // Draw grid
            this.poseCtx.strokeStyle = "rgba(255, 255, 255, 0.04)";
            this.poseCtx.lineWidth = 1;
            for (let x = 0; x < w; x += 30) {
                this.poseCtx.beginPath();
                this.poseCtx.moveTo(x, 0);
                this.poseCtx.lineTo(x, h);
                this.poseCtx.stroke();
            }
            for (let y = 0; y < h; y += 30) {
                this.poseCtx.beginPath();
                this.poseCtx.moveTo(0, y);
                this.poseCtx.lineTo(w, y);
                this.poseCtx.stroke();
            }

            // Get frame
            const frame = this.currentPoseFrames[this.currentPoseFrameIdx];
            this.currentPoseFrameIdx = (this.currentPoseFrameIdx + 1) % this.currentPoseFrames.length;

            const toScreen = (pt) => ({ x: pt.x * w, y: pt.y * h });

            // Bone connectivity pairs
            const bones = [
                ['head', 'neck'],
                ['neck', 'l_shoulder'], ['neck', 'r_shoulder'],
                ['l_shoulder', 'l_elbow'], ['l_elbow', 'l_wrist'],
                ['r_shoulder', 'r_elbow'], ['r_elbow', 'r_wrist'],
                ['neck', 'hip'],
                ['hip', 'l_hip'], ['hip', 'r_hip'],
                ['l_hip', 'l_knee'], ['l_knee', 'l_ankle'],
                ['r_hip', 'r_knee'], ['r_knee', 'r_ankle']
            ];

            // Color scheme based on class
            let boneColor = "#38bdf8";
            let jointColor = "#06b6d4";
            if (this.currentClass === "HeadBanging") {
                boneColor = "#f87171";
                jointColor = "#ef4444";
            } else if (this.currentClass === "ArmFlapping") {
                boneColor = "#fbbf24";
                jointColor = "#f59e0b";
            }

            // Draw Bones
            this.poseCtx.lineWidth = 3.5;
            this.poseCtx.strokeStyle = boneColor;
            this.poseCtx.shadowColor = boneColor;
            this.poseCtx.shadowBlur = 10;

            bones.forEach(([j1, j2]) => {
                if (frame[j1] && frame[j2]) {
                    const p1 = toScreen(frame[j1]);
                    const p2 = toScreen(frame[j2]);
                    this.poseCtx.beginPath();
                    this.poseCtx.moveTo(p1.x, p1.y);
                    this.poseCtx.lineTo(p2.x, p2.y);
                    this.poseCtx.stroke();
                }
            });

            // Draw Joint Nodes
            for (const k in frame) {
                const pt = toScreen(frame[k]);
                this.poseCtx.beginPath();
                const radius = (k === 'head' || k === 'l_wrist' || k === 'r_wrist') ? 6.5 : 4.5;
                this.poseCtx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
                this.poseCtx.fillStyle = jointColor;
                this.poseCtx.shadowColor = jointColor;
                this.poseCtx.shadowBlur = 12;
                this.poseCtx.fill();
            }

            // Watermark / Indicator
            this.poseCtx.shadowBlur = 0;
            this.poseCtx.fillStyle = "rgba(255, 255, 255, 0.4)";
            this.poseCtx.font = "11px 'JetBrains Mono', monospace";
            this.poseCtx.fillText(`SYNTHETIC TIMEGAN • 30 FPS • ${this.currentClass.toUpperCase()}`, 12, 22);

            setTimeout(() => {
                this.poseAnimationId = requestAnimationFrame(render);
            }, 33); // ~30 FPS
        };
        render();
    }
}

// Global visualizer singleton
window.visualizer = new VisualizerEngine();
