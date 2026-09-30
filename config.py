import os

# ---------------------------------------------------------------------------
# Force all model download & cache directories into d:\SIH\.cache
# Ensures ZERO disk usage on local Disk C.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TORCH_HOME"] = os.path.join(CACHE_DIR, "torch")
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["MPLCONFIGDIR"] = os.path.join(CACHE_DIR, "matplotlib")

import torch

class Config:
    # Project Paths
    BASE_DIR = BASE_DIR
    CACHE_DIR = CACHE_DIR
    DATA_DIR = os.path.join(BASE_DIR, "Data") if os.path.exists(os.path.join(BASE_DIR, "Data")) else os.path.join(BASE_DIR, "data")
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train_images")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    GRADCAM_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "gradcam_examples")
    SHAP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "xai_report")
    SHAP_EXAMPLES_DIR = os.path.join(SHAP_OUTPUT_DIR, "shap_examples")
    SHAP_CONFIDENCE_DIR = os.path.join(SHAP_OUTPUT_DIR, "confidence_panels")
    SHAP_CACHE_DIR = os.path.join(SHAP_OUTPUT_DIR, ".cache")

    # Multi-Dataset Paths (APTOS, IDRiD, DRIVE)
    IDRID_DIR = os.path.join(DATA_DIR, "iDRiD")
    IDRID_TRAIN_CSV = os.path.join(DATA_DIR, "idrid_train.csv")
    IDRID_TEST_CSV = os.path.join(DATA_DIR, "idrid_test.csv")
    COMBINED_TRAIN_CSV = os.path.join(DATA_DIR, "combined_train.csv")
    DRIVE_DIR = os.path.join(DATA_DIR, "DRIVE")
    VESSEL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "vessel_unet.pth")
    DATASET_MODE = "combined"  # Options: 'aptos', 'idrid', 'combined'

    NUM_CLASSES = 5
    CLASS_LABELS = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR"
    }
    VAL_SIZE = 0.15
    SEED = 42

    # Image Preprocessing & Model Setup
    # High-resolution 384x384 for fine micro-lesion detection
    IMG_SIZE = 384
    MODEL_NAME = "swin_tiny_patch4_window7_224"  # backbone dynamically scaled to 384
    FALLBACK_MODEL_NAME = "tf_efficientnet_b3_ns"
    PRETRAINED = True

    # Training Parameters
    EPOCHS = 30
    PATIENCE = 10
    FREEZE_BACKBONE_EPOCHS = 2
    BATCH_SIZE = 12  # VRAM-optimized for 6GB GPU at 384x384
    GRAD_ACCUM_STEPS = 1
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    USE_AMP = True  # Automatic Mixed Precision for fast CUDA training
    USE_REGRESSION = True  # Ordinal Regression for QWK optimization
    BALANCED_SAMPLING = True  # WeightedRandomSampler for class balance

    # Hardware & System
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    @classmethod
    def create_dirs(cls):
        r"""Ensure all required directories exist inside d:\SIH."""
        for path in [cls.CACHE_DIR, cls.DATA_DIR, cls.TRAIN_IMAGES_DIR, cls.PROCESSED_DIR,
                     cls.CHECKPOINT_DIR, cls.OUTPUT_DIR, cls.GRADCAM_OUTPUT_DIR,
                     cls.SHAP_OUTPUT_DIR, cls.SHAP_EXAMPLES_DIR, cls.SHAP_CONFIDENCE_DIR, cls.SHAP_CACHE_DIR]:
            os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    Config.create_dirs()
    print("Configuration initialized.")
    print(f"Device: {Config.DEVICE}")
    print(f"Model Backbone: {Config.MODEL_NAME}")
    print(f"Data Dir: {Config.DATA_DIR}")
    print(f"Cache Dir: {Config.CACHE_DIR}")

