import sys
import os
from pathlib import Path

# Add project root (d:\SIH) to sys.path so infer, config, model, shap_report can be imported directly
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from infer import load_inference_model

print("[INFO] Initializing Diabetic Retinopathy model singleton at server startup...")
# Loads the trained model weights once when the backend module is imported
model = load_inference_model()
print("[INFO] Model loaded successfully and cached for inference requests.")
