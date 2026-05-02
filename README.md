# Degraded Photo Detection & Enhancement

A full-stack AI desktop application that scans a photo gallery, flags degraded images using a two-gate hybrid pipeline, and restores them using multi-stage classical + deep learning enhancement. Ships as a native macOS `.dmg` installer built with Electron.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Gate 1 — Laplacian Blur Filter](#gate-1--laplacian-blur-filter)
- [Gate 2 — MobileNetV2 + Spatial Attention CNN](#gate-2--mobilenetv2--spatial-attention-cnn)
- [Enhancement Pipeline — Gate 1 (Blur)](#enhancement-pipeline--gate-1-blur)
- [Enhancement Pipeline — Gate 2 (CNN-Flagged)](#enhancement-pipeline--gate-2-cnn-flagged)
- [Training](#training)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Building the Desktop Installer](#building-the-desktop-installer)
- [Deployment](#deployment)

---

## Overview

Photos are flagged in two passes:

1. **Gate 1 (fast)** — Laplacian variance: pure math, no model, runs in microseconds. Catches severe blur.
2. **Gate 2 (deep)** — MobileNetV2 with a custom Spatial Attention Block. Catches subtle degradation: sensor noise, poor exposure, accidental composition.

Flagged images can then be enhanced on-demand via multi-stage pipelines that combine classical signal processing (Richardson-Lucy deconvolution, NLM denoising, CLAHE) with ESRGAN generative super-resolution.

The entire application — React frontend + FastAPI backend + AI models — runs as a single native macOS desktop app launched from one command.

---

## Architecture

```
User (Desktop Window / Electron)
    │
    ▼
FastAPI Server (server.py)  ←  serves React build at /
    │
    ├── GET  /api/browse ────────────► AppleScript folder picker (macOS native)
    │
    ├── POST /api/scan ──────────────► clean_my_drive.py
    │                                      │
    │                                      ├── Gate 1: Laplacian variance
    │                                      │     < 100 → "Severe Blur"
    │                                      │
    │                                      └── Gate 2: MobileNetV2 CNN
    │                                            class 1 → "Degraded Composition"
    │
    ├── POST /api/enhance ──────────► src/enhance.py
    │                                      │
    │                                      ├── Blur → 5-stage pipeline
    │                                      │     R-L → Bilateral → CLAHE → ESRGAN → Unsharp
    │                                      │
    │                                      └── CNN → 6-stage pipeline
    │                                            NLM → CLAHE → AWB → ESRGAN → Bilateral → Unsharp
    │
    └── GET  /api/image ─────────────► FileResponse (serves image bytes to browser)
```

---

## Tech Stack

| Layer | Library | Purpose |
|---|---|---|
| Desktop Shell | Electron | Wraps the web UI as a native macOS/Windows app |
| Deep Learning | TensorFlow / Keras | CNN training + ESRGAN inference |
| Model Hub | TensorFlow Hub | Hosts ESRGAN SavedModel |
| Image Processing | OpenCV (headless) | Filtering, colour ops, Laplacian, NLM, CLAHE |
| Signal Processing | SciPy | FFT convolution for Richardson-Lucy deconvolution |
| Data Augmentation | Albumentations | Synthetic degradation (blur, noise, exposure) |
| Numerical | NumPy | Array ops, Wiener fallback, grey-world AWB |
| Backend | FastAPI + Uvicorn | Async REST API with Pydantic validation |
| Frontend | React + Vite | UI with before/after toggle, staggered animations |
| Installer | electron-builder | Produces `.dmg` (macOS) and `.exe` (Windows) |

---

## Gate 1 — Laplacian Blur Filter

**File:** `clean_my_drive.py`

```python
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
flagged = laplacian_var < 100.0
```

The Laplacian is the second spatial derivative of the image. Sharp images have rapid intensity transitions (high variance); blurry images have smooth gradients (low variance). The threshold of `100.0` is tunable via `--blur_threshold`.

Every image passes Gate 1 first. Only those that pass (are sharp) continue to the expensive CNN — keeping the system fast.

---

## Gate 2 — MobileNetV2 + Spatial Attention CNN

**File:** `src/train.py`

### Model Architecture

```
Input 224×224×3
    ↓
MobileNetV2 backbone (first 100 layers frozen — ImageNet weights)
    ↓
Spatial Attention Block:
    Conv2D(1, kernel_size=1, activation='sigmoid') → importance map
    Multiply([features, importance_map])
    ↓
GlobalAveragePooling2D
    ↓
Dense(1, activation='sigmoid') → P(degraded)
```

**MobileNetV2** uses depthwise separable convolutions (8–9× fewer operations than standard convs) making it fast enough for per-image batch inference.

**Spatial Attention** collapses feature channels to a single H×W importance map. The network learns to focus on spatially localized degradation (motion blur in one region, noise in shadows).

### Loss Function — Binary Focal Cross-Entropy

```
FL = -(1 - p)^γ · log(p)    γ=2.0, α=0.25
```

`(1-p)^γ` down-weights easy examples (model is already confident) and amplifies gradient on hard borderline cases (subtle degradation). `α=0.25` balances the pristine/degraded class ratio.

### Optimizer — AdamW

AdamW decouples L2 weight decay from the adaptive gradient update, producing better generalisation than standard Adam with L2 regularisation.

---

## Enhancement Pipeline — Gate 1 (Blur)

**File:** `src/enhance.py` → `enhance_blurry_image()`

| Stage | Method | Purpose |
|---|---|---|
| 1 | Richardson-Lucy deconvolution (25 iter) | Mathematically inverts the blur convolution |
| 2 | Bilateral filter | Suppresses ringing artefacts from deconvolution |
| 3 | CLAHE on LAB L-channel | Restores local contrast compressed by blur |
| 4 | ESRGAN 4× super-resolution | Hallucinates high-frequency texture |
| 5 | Unsharp mask (strength 1.5) | Locks in recovered sharpness |

### Richardson-Lucy Algorithm

Blur is modelled as: `blurry = original ⊛ PSF`

R-L iteratively inverts this:
```
estimate_{n+1} = estimate_n × (PSF* ⊛ (blurry / (PSF ⊛ estimate_n)))
```

**Auto PSF estimation** — analyses gradient orientation histogram. If >35% of strong edges share a direction → motion blur PSF. Otherwise → Gaussian PSF.

Wiener deconvolution is available as a fallback if SciPy is unavailable.

---

## Enhancement Pipeline — Gate 2 (CNN-Flagged)

**File:** `src/enhance.py` → `enhance_degraded_composition()`

| Stage | Method | Purpose |
|---|---|---|
| 1 | NLM denoising | Removes sensor noise while preserving edges |
| 2 | CLAHE on LAB L-channel | Boosts local contrast without clipping highlights |
| 3 | Auto white balance (grey-world) | Corrects colour casts from bad lighting |
| 4 | ESRGAN 4× super-resolution | Perceptual quality and texture recovery |
| 5 | Bilateral filter | Cleans ESRGAN resize artefacts |
| 6 | Unsharp mask (strength 1.4) | Final edge lift to compensate NLM softening |

### Grey-World White Balance

```python
avg_all = (mean_B + mean_G + mean_R) / 3.0
channel *= (avg_all / channel_mean)
```

Assumes a natural scene has equal average luminance across all three channels. Scales each channel to match the global mean, removing orange/blue/green colour casts.

---

## Training

**File:** `src/data_prep.py` — synthetic degradation via Albumentations:

| Augmentation | Probability | Simulates |
|---|---|---|
| `AdvancedBlur` | 40% | Optical defocus / generalised Gaussian aberration |
| `MotionBlur` | 40% | Camera shake |
| `ISONoise` | 40% | Photon shot noise + read noise |
| `RandomBrightnessContrast` | 40% | Exposure error |
| `RandomResizedCrop` | 30% | Accidental composition / partial subject |

Training uses `tf.data.Dataset` with `AUTOTUNE` parallelism and `prefetch` to prevent GPU starvation.

To retrain:
```bash
# Expects data/raw/train/{good,bad}/ and data/raw/val/{good,bad}/
cd src
python3 train.py
```

---

## Project Structure

```
DegradedPhotoDetection/
├── electron/
│   └── main.js                # Electron entry: installs deps, spawns server, opens window
├── src/
│   ├── train.py               # Model architecture + training loop
│   ├── data_prep.py           # tf.data pipeline + Albumentations augmentation
│   └── enhance.py             # Multi-stage enhancement pipelines
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # React UI
│   │   └── App.css            # Dark theme + animations
│   ├── dist/                  # Built React files (served by FastAPI)
│   └── package.json
├── models/
│   └── weights/
│       └── best_mobilenetv2.keras
├── server.py                  # FastAPI REST API + serves frontend/dist at /
├── clean_my_drive.py          # Scan engine (Gate 1 + Gate 2 inference)
├── requirements.txt           # Python dependencies
├── package.json               # Electron + electron-builder config
├── setup.sh                   # One-command developer setup
└── enhanced_output/           # Enhanced images saved here
```

---

## Installation

### Option A — Desktop App (Recommended)

> Requires Python 3 to be installed. Everything else is automatic.

1. Download `Degraded Photo Detection-1.0.0.dmg`
2. Open the `.dmg` and drag the app to your **Applications** folder
3. Open the app
4. On first launch, Python dependencies install automatically (~2 minutes)
5. Every launch after that opens in ~3 seconds

No terminal, no commands.

---

### Option B — Developer Setup (from source)

Requires Python 3, Node.js, and npm.

```bash
git clone <repo-url>
cd DegradedPhotoDetection
chmod +x setup.sh && ./setup.sh
```

`setup.sh` creates the Python venv, installs all pip dependencies, installs Node modules, and builds the React frontend automatically.

Then run the app:
```bash
npm start
```

---

## Running the App

### Desktop (Electron window)

```bash
npm start
```

Electron launches, boots the FastAPI server in the background, and opens the full UI in a native window at `http://127.0.0.1:8000`.

### Browser only (dev mode)

```bash
# Terminal 1 — backend
source venv/bin/activate
python3 server.py
# → http://localhost:8000

# Terminal 2 — frontend hot-reload
cd frontend
npm run dev
# → http://localhost:5173
```

### How to use the app

1. Click **Browse** to pick a photo folder using the native macOS folder picker
2. Click **Scan Gallery** — the AI scans all `.jpg` / `.png` files in two passes
3. Flagged photos appear with a reason tag:
   - `Blur` — failed Laplacian Gate (Gate 1)
   - `Degraded Composition` — failed MobileNetV2 Gate (Gate 2)
4. Click **Enhance** on any card to run the restoration pipeline
5. Toggle **Original / Enhanced** to compare before and after
6. Click **Save** to download the restored image

---

## Building the Desktop Installer

Produces a `.dmg` (macOS) that anyone can install by dragging to Applications.

```bash
npm run dist
```

Output:
```
dist/
└── Degraded Photo Detection-1.0.0.dmg
```

For Windows:
```bash
npm run dist:win
# → dist/Degraded Photo Detection Setup 1.0.0.exe
```

---

## Deployment

The app is designed as a local desktop tool — it reads directly from the user's filesystem and uses the macOS native folder picker. This is intentional: no files are uploaded to external servers.

**Containerised local deployment (Docker):**
```bash
docker build -t degraded-photo-app .
docker run -p 8000:8000 -v ~/Photos:/photos degraded-photo-app
```

**Cloud demo (Hugging Face Spaces):** Replace the React UI with a Gradio interface and push to a Space for a public URL with free GPU access.
