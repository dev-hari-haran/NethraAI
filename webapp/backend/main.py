import os
import sys
import uuid
import base64
import tempfile
from pathlib import Path
import numpy as np

# Ensure project root (d:\SIH) and backend directory are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = Path(__file__).resolve().parent

for p in [str(ROOT_DIR), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config import Config
from preprocess import preprocess_single_image
from shap_report import generate_confidence_panel, compute_image_shap, SHAPPredictWrapper

try:
    from webapp.backend import model_state
except ImportError:
    import model_state

# Ensure all required project directories exist
Config.create_dirs()

# Ensure dedicated tmp directory exists
TMP_DIR = os.path.join(Config.BASE_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

app = FastAPI(
    title="Diabetic Retinopathy XAI API",
    description="FastAPI service for APTOS DR classification and SHAP visual explanations.",
    version="1.0.0"
)

# Enable permissive CORS for local development and demo access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@app.get("/health")
def health_check():
    """Health check endpoint to verify backend and model status."""
    return {
        "status": "online",
        "model_loaded": model_state.model is not None,
        "device": Config.DEVICE,
        "model_backbone": Config.MODEL_NAME,
        "num_classes": Config.NUM_CLASSES,
        "classes": Config.CLASS_LABELS
    }


@app.get("/sample-image/{filename}")
async def get_sample_image(filename: str):
    """Provides sample fundus images from Data directory for quick UI testing."""
    safe_name = os.path.basename(filename)
    candidates = [
        os.path.join(Config.DATA_DIR, safe_name),
        os.path.join(Config.TRAIN_IMAGES_DIR, safe_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="Sample image not found.")


@app.post("/predict")
async def predict_retina(file: UploadFile = File(...)):
    """
    Accepts a retinal fundus image upload, performs DR classification,
    generates a dual-panel SHAP attribution + confidence figure, and
    returns predictions with a base64-encoded visual explanation.
    """
    # 1. Validate file extension
    filename = file.filename or "upload.png"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # 2. Save uploaded bytes to a temporary file
    temp_upload_name = f"upload_{uuid.uuid4().hex[:10]}{ext}"
    temp_upload_path = os.path.join(TMP_DIR, temp_upload_name)

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with open(temp_upload_path, "wb") as f:
            f.write(contents)

        # 3. Validate image readability and preprocess
        try:
            img_rgb = preprocess_single_image(
                temp_upload_path,
                target_size=Config.IMG_SIZE,
                apply_ben_graham=True
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process fundus image. File may be corrupt or not a valid image. Error: {str(e)}"
            )

        # 4. Generate dual-panel SHAP explanation
        unique_panel_name = f"conf_panel_{uuid.uuid4().hex[:8]}.png"
        try:
            panel_path = generate_confidence_panel(
                model_state.model,
                temp_upload_path,
                output_filename=unique_panel_name
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating SHAP explanation panel: {str(e)}"
            )

        # 5. Retrieve probability distribution & predicted class
        # (Hits disk cache instantly since generate_confidence_panel already computed it)
        wrapper = SHAPPredictWrapper(model_state.model)
        _, probs = compute_image_shap(wrapper, img_rgb)

        pred_class = int(probs.argmax())
        pred_label = Config.CLASS_LABELS.get(pred_class, f"Class {pred_class}")
        confidence = float(probs[pred_class])
        all_probs = [float(p) for p in probs]

        # 6. Read and base64-encode the generated SHAP panel PNG
        if not os.path.exists(panel_path):
            raise HTTPException(
                status_code=500,
                detail="SHAP explanation panel was not found on disk after generation."
            )

        with open(panel_path, "rb") as img_file:
            shap_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        # 7. Return structured response
        return JSONResponse(content={
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "all_probs": [round(p, 4) for p in all_probs],
            "shap_panel_base64": shap_base64
        })

    finally:
        # Clean up temporary uploaded file
        if os.path.exists(temp_upload_path):
            try:
                os.remove(temp_upload_path)
            except OSError:
                pass


@app.post("/api/vessel-segmentation")
async def extract_retinal_vessels(file: UploadFile = File(...)):
    """
    Endpoint for DRIVE-based retinal blood vessel extraction and vascular density metrics.
    """
    import cv2
    from vessel_segmentor import get_vessel_segmentor

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    temp_id = f"vessel_{uuid.uuid4().hex[:8]}{ext}"
    temp_upload_path = os.path.join(TMP_DIR, temp_id)

    try:
        content = await file.read()
        with open(temp_upload_path, "wb") as f:
            f.write(content)

        img_bgr = cv2.imread(temp_upload_path)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not read uploaded fundus image.")

        segmentor = get_vessel_segmentor()
        _, binary_mask, overlay, metrics = segmentor.segment_vessels(img_bgr)

        # Encode overlay and mask as PNG base64
        _, overlay_buf = cv2.imencode(".png", overlay)
        _, mask_buf = cv2.imencode(".png", binary_mask)

        return JSONResponse(content={
            "vessel_overlay_base64": base64.b64encode(overlay_buf).decode("utf-8"),
            "vessel_mask_base64": base64.b64encode(mask_buf).decode("utf-8"),
            "metrics": metrics
        })

    finally:
        if os.path.exists(temp_upload_path):
            try:
                os.remove(temp_upload_path)
            except OSError:
                pass



# Mount frontend static files directly at root so index.html, style.css, and app.js are all resolved
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
