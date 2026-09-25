# Diabetic Retinopathy Diagnostic Web App (Model + SHAP XAI)

A web interface and FastAPI backend providing automated Diabetic Retinopathy (DR) grading powered by a fine-tuned Swin Transformer backbone and Explainable AI (XAI) using SHAP (SHapley Additive exPlanations).

---

## 1. Quick Start

### Step 1: Install Dependencies
Install the required web packages into the virtual environment:

```powershell
# From the project root (D:\SIH)
d:\SIH\.venv\Scripts\python.exe -m pip install fastapi uvicorn python-multipart
```

*(All other machine learning, PyTorch, timm, and SHAP libraries are already installed in `.venv`)*

---

### Step 2: Start the FastAPI Backend Server
Run the Uvicorn ASGI server from the `d:\SIH` directory:

```powershell
# PowerShell command:
d:\SIH\.venv\Scripts\python.exe -m uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

When started, the model checkpoint (`checkpoints/best_model.pth`) will be loaded **once** at startup via `webapp/backend/model_state.py`.

---

### Step 3: Open the Web Application
You can open the web application in any modern web browser in either of two ways:

1. **Directly via the FastAPI Server (Recommended):**
   ```
   http://127.0.0.1:8000/
   ```
   *(The backend mounts and serves `webapp/frontend/index.html` automatically)*

2. **Or open the file directly in your browser:**
   ```
   d:\SIH\webapp\frontend\index.html
   ```
   *(CORS is enabled on the backend, allowing local `file://` or Live Server origins to communicate seamlessly with `http://localhost:8000/predict`)*

---

## 2. Testing the Application

### Option A: Via the Web UI
1. Open `http://127.0.0.1:8000/` in your browser.
2. Drag and drop any retinal fundus image (`.png`, `.jpg`, `.jpeg`, `.tiff`) into the upload zone, or click one of the **Quick Test Fundus Samples** buttons (`Sample M`, `Sample N1`, `Sample P1`, `IDRiD 02`).
3. Click **"Run Diagnostic & SHAP Analysis"**.
4. The system will process the image:
   - Apply Ben Graham local color enhancement.
   - Run Swin Transformer inference with Test-Time Augmentation (TTA).
   - Generate the dual-panel SHAP attribution map.
5. The diagnostic panel displays:
   - Clinical severity assessment (Classes 0–4 with color-coded badge).
   - Calibrated probability distribution across all 5 classes.
   - Dual-panel SHAP visual explanation (Retina + SHAP heatmap on the left, confidence chart on the right).
   - Click **"Zoom View"** or click on the SHAP image to open the high-resolution lightbox.

> **Performance Note on Caching:** The first inference for a new image takes ~5–10 seconds because SHAP evaluates 300 image perturbations. Repeat requests for the same image are **near-instant** (<100ms) because `compute_image_shap` disk-caches the explanation in `outputs/xai_report/.cache/`.

---

### Option B: Via Command Line (`curl` or PowerShell)
To test the `/predict` API endpoint directly from PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" `
  -H "accept: application/json" `
  -H "Content-Type: multipart/form-data" `
  -F "file=@D:\SIH\Data\samplem.png"
```

Or test backend health:
```powershell
curl.exe "http://127.0.0.1:8000/health"
```

Expected JSON response from `/predict`:
```json
{
  "predicted_class": 2,
  "predicted_label": "Moderate DR",
  "confidence": 0.874,
  "all_probs": [0.012, 0.051, 0.874, 0.048, 0.015],
  "shap_panel_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

---

## 3. Directory Architecture

```
d:\SIH\
├── checkpoints/
│   └── best_model.pth          # Trained Swin-Tiny checkpoint
├── Data/                       # Dataset & test sample images
├── config.py                   # Central paths and training hyperparameters
├── infer.py                    # load_inference_model() and inference pipeline
├── preprocess.py               # preprocess_single_image() Ben Graham method
├── shap_report.py              # Existing SHAP engine (generate_confidence_panel, etc.)
└── webapp/
    ├── README.md               # This guide
    ├── backend/
    │   ├── model_state.py      # Singleton: loads checkpoint once at import
    │   └── main.py             # FastAPI app with /predict, /health, CORS, and static mount
    └── frontend/
        ├── index.html          # Clinical diagnostic dashboard
        ├── style.css           # Modern medical dark theme & micro-animations
        └── app.js              # Fetch client, drag & drop, sample loader, and SHAP renderer
```

---

## 4. Key Architectural Safeguards

- **Single Model Load**: `model_state.py` loads `load_inference_model()` as a module singleton so the 330MB PyTorch weights are never reloaded per request.
- **Zero Local Disk C Clutter**: All model outputs and caches are saved within `d:\SIH` (`outputs/xai_report/confidence_panels/` and `.cache/`).
- **Disk Caching**: SHAP explanations are cached by MD5 hash in `Config.SHAP_CACHE_DIR`, ensuring presentation-ready instant recalls for tested images.
- **Dual Serving**: The frontend can be opened directly from disk or hosted automatically by FastAPI.
