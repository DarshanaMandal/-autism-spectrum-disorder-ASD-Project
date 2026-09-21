# MultiSense: Complete GitHub Setup & Cloud Hosting Deployment Guide

This guide provides step-by-step instructions to:
1. Initialize Git and push your project to GitHub cleanly without hitting file size limits.
2. Deploy the full-stack system (FastAPI Backend + Interactive Web UI) to the cloud for free (**Render / Railway / Docker**).
3. Optionally host the frontend on **Vercel / Netlify**.

---

## 1. Step-by-Step: Pushing to GitHub Cleanly

### 1.1 Why `.gitignore` is Critical for this Project
* Raw video folders (`data/raw/`, `data/sequences/`, `data/processed/`) total **~550 MB**.
* GitHub enforces a **100 MB per file limit** and will reject the push if large videos or intermediate arrays are included.
* Our `.gitignore` automatically ignores raw videos while **safely preserving**:
  - All 4 trained model checkpoints (`MultiSense/models/*.pkl`, `gan_checkpoint.pth` — total only ~11 MB)
  - Extracted feature datasets (`MultiSense/data/features/` — 2.6 MB)
  - Dataset annotations (`MultiSense/data/ssbd_annotations.json` — 56 KB)
  - High-resolution evaluation charts (`MultiSense/results/*.png`, `*.csv`)
  - Demo video clips for the Web UI (`MultiSense/frontend/samples/*.mp4` — 1 MB)

---

### 1.2 Terminal Commands to Push to GitHub

Open your Mac terminal in the project root:

```bash
cd /Users/shreyashgautam/Desktop/darshana/authism

# 1. Initialize git repository
git init

# 2. Verify gitignore is active (should show ~20-30 MB tracked files instead of 600 MB)
git status

# 3. Stage all source code, models, API, frontend & documentation
git add .

# 4. Create your initial commit
git commit -m "feat: MultiSense Multimodal ASD Detection Platform (FastAPI + TimeGAN + Modern UI)"
```

### 1.3 Link to Your GitHub Account
1. Go to [GitHub.com](https://github.com/) and click **New Repository**.
2. Name it: `multisense-asd` (or any name you prefer).
3. Keep it **Public** (or Private), and **DO NOT** initialize with a README (we already have one).
4. Copy your repository URL and run:

```bash
# Rename branch to main
git branch -M main

# Add remote origin (replace with your GitHub username)
git remote add origin https://github.com/<YOUR-USERNAME>/multisense-asd.git

# Push code to GitHub
git push -u origin main
```

---

## 2. Recommended Hosting Method: Render.com (100% Free & All-in-One)

Render is the simplest cloud hosting platform. It will build and serve **both the backend API and the interactive frontend dashboard** on a single public URL (e.g. `https://multisense-ai.onrender.com`).

### 2.1 Step-by-Step Deployment on Render:
1. Sign up / Log in to [Render.com](https://render.com) using your GitHub account.
2. On the Render dashboard, click **New +** (top right) and select **Web Service**.
3. Select **Build and deploy from a Git repository**, click **Next**, and choose your `multisense-asd` repository.
4. Fill in the deployment form:
   * **Name**: `multisense-ai` (or any unique name)
   * **Region**: Choose closest to you (e.g., *Singapore* or *Frankfurt*)
   * **Branch**: `main`
   * **Root Directory**: Leave blank (root of repository)
   * **Runtime**: `Python 3`
   * **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   * **Start Command**:
     ```bash
     uvicorn MultiSense.api.main:app --host 0.0.0.0 --port $PORT
     ```
   * **Instance Type**: Select **Free**
5. Click **Create Web Service**.

### 2.2 What Happens Next:
* Render installs dependencies from `requirements.txt`.
* The server boots up and assigns a live public URL: `https://multisense-ai.onrender.com`.
* **Access Points**:
  - Web UI Dashboard: `https://multisense-ai.onrender.com/`
  - Interactive Swagger Docs: `https://multisense-ai.onrender.com/docs`
  - Health Check: `https://multisense-ai.onrender.com/health`

---

## 3. Alternative Hosting: Railway.app (Fast 1-Click Deploy)

1. Sign up at [Railway.app](https://railway.app) with GitHub.
2. Click **New Project** ➔ **Deploy from GitHub repo**.
3. Select `multisense-asd`.
4. Railway automatically detects the [`Procfile`](file:///Users/shreyashgautam/Desktop/darshana/authism/Procfile) and deploys immediately!
5. In your project settings, click **Generate Domain** to get your live `.up.railway.app` URL.

---

## 4. Decoupled Frontend Hosting (Vercel / Netlify)

If you prefer to host the frontend statically on **Vercel** or **Netlify** while keeping the FastAPI backend on Render:

1. In Vercel, import the repository and set:
   * **Root Directory**: `MultiSense/frontend`
   * **Framework Preset**: `Other`
2. Deploy.
3. Once live, open the Vercel URL in your browser:
   * Click the **`⚙️ Host Settings`** button in the top navigation bar.
   * Enter your deployed Render backend URL (e.g. `https://multisense-ai.onrender.com`).
   * Click **Save & Connect**. The frontend will immediately route all video predictions, telemetry simulations, and GAN generations to your remote cloud backend!

---

## 5. Docker Deployment (Local, VPS, AWS EC2, DigitalOcean)

The project includes a production [`Dockerfile`](file:///Users/shreyashgautam/Desktop/darshana/authism/Dockerfile) and [`docker-compose.yml`](file:///Users/shreyashgautam/Desktop/darshana/authism/docker-compose.yml).

### Run with Docker:
```bash
# Build the image
docker build -t multisense-app .

# Run container on port 8000
docker run -p 8000:8000 multisense-app
```

### Or using Docker Compose:
```bash
docker-compose up --build
```
The application will be live at `http://localhost:8000/`.

---

## 6. Pre-Flight Hosting Checklist

| Item | Status | Purpose |
| :--- | :---: | :--- |
| [`.gitignore`](file:///Users/shreyashgautam/Desktop/darshana/authism/.gitignore) | Verified | Excludes ~550 MB raw videos so Git repo is lightweight (<25 MB) |
| [`requirements.txt`](file:///Users/shreyashgautam/Desktop/darshana/authism/requirements.txt) | Verified | Uses `opencv-python-headless` to avoid `libGL.so.1` Linux cloud errors |
| [`Procfile`](file:///Users/shreyashgautam/Desktop/darshana/authism/Procfile) | Verified | Binds dynamically to cloud provider port `$PORT` |
| [`render.yaml`](file:///Users/shreyashgautam/Desktop/darshana/authism/render.yaml) | Verified | Enables 1-click Blueprints deployment |
| [`MultiSense/models/`](file:///Users/shreyashgautam/Desktop/darshana/authism/MultiSense/models) | Verified | All 4 models total 11 MB (<100 MB limit, no Git LFS needed) |
| CORS Configuration | Verified | Middleware enabled in FastAPI to permit cross-origin calls |
| Automated Tests | 100% PASS | Verified via `python3 MultiSense/api/test_api.py` |
