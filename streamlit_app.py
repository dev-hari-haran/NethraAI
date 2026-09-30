import os
import sys
import uuid
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import streamlit as st

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config
from infer import load_inference_model, predict_single_image
from vessel_segmentor import get_vessel_segmentor
from shap_report import generate_confidence_panel, compute_image_shap, SHAPPredictWrapper
from preprocess import preprocess_single_image

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Theming
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NethraAI — Explainable Retinal Diagnostic Suite",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Medical Aesthetic Styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .badge-card {
        padding: 16px 20px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-container {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #0284c7;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached Model Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Swin Transformer & DRIVE U-Net...")
def load_all_models():
    """Loads and caches both the DR classification model and the vessel segmentor."""
    Config.create_dirs()
    clf_model = load_inference_model()
    vessel_seg = get_vessel_segmentor()
    return clf_model, vessel_seg

try:
    model, vessel_segmentor = load_all_models()
    models_ready = True
except Exception as e:
    models_ready = False
    load_error = str(e)

# -----------------------------------------------------------------------------
# Clinical Grade Color & Recommendation Mappings
# -----------------------------------------------------------------------------
SEVERITY_META = {
    0: {
        "title": "Class 0 — No Diabetic Retinopathy",
        "color": "#10b981", # Emerald
        "urgency": "Low Risk",
        "recommendation": "Normal retinal fundus. Routine annual screening recommended.",
    },
    1: {
        "title": "Class 1 — Mild Non-Proliferative DR",
        "color": "#eab308", # Amber
        "urgency": "Mild Risk",
        "recommendation": "Microaneurysms detected. Comprehensive eye exam within 6–12 months.",
    },
    2: {
        "title": "Class 2 — Moderate Non-Proliferative DR",
        "color": "#f97316", # Orange
        "urgency": "Moderate Risk",
        "recommendation": "Hemorrhages, hard exudates, or cotton wool spots. Refer to ophthalmologist within 2–4 months.",
    },
    3: {
        "title": "Class 3 — Severe Non-Proliferative DR",
        "color": "#ef4444", # Red
        "urgency": "High Risk",
        "recommendation": "Extensive intraretinal hemorrhages or venous beading. Urgent ophthalmology consultation within 2–4 weeks.",
    },
    4: {
        "title": "Class 4 — Proliferative Diabetic Retinopathy (PDR)",
        "color": "#991b1b", # Dark Crimson
        "urgency": "Critical / Sight-Threatening",
        "recommendation": "Active neovascularization or vitreous hemorrhage. Immediate vitreoretinal intervention required (anti-VEGF/PRP laser).",
    }
}

# -----------------------------------------------------------------------------
# Sidebar: System Controls & Sample Selection
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/visible.png", width=64)
    st.title("NethraAI Suite")
    st.caption("AI-Powered Ophthalmology Assistant")

    st.markdown("---")
    st.subheader("System Status")
    if models_ready:
        st.success(f"Models Ready ({Config.DEVICE.upper()})")
        st.caption(f"**Backbone:** Swin-Tiny (384x384 FP16)")
        st.caption(f"**Vessel Extractor:** DRIVE U-Net (32-base)")
    else:
        st.error(f"Failed to load models: {load_error}")

    st.markdown("---")
    st.subheader("Quick Test Samples")
    sample_options = {
        "Upload Custom Image": None,
        "Sample M (Mild DR)": os.path.join(Config.DATA_DIR, "samplem.png"),
        "Sample N1 (Normal / No DR)": os.path.join(Config.DATA_DIR, "samplen1.png"),
        "Sample P1 (Proliferative DR)": os.path.join(Config.DATA_DIR, "samplep1.png"),
        "Sample P2 (Severe / PDR)": os.path.join(Config.DATA_DIR, "samplep2.png"),
        "Sample P3 (Proliferative)": os.path.join(Config.DATA_DIR, "samplep3.png"),
        "IDRiD 02 (Clinical Sample)": os.path.join(Config.DATA_DIR, "IDRiD_02.jpg"),
        "IDRiD 07 (Clinical Sample)": os.path.join(Config.DATA_DIR, "IDRiD_07.jpg"),
    }
    selected_sample = st.selectbox("Select demo fundus image:", list(sample_options.keys()))

    st.markdown("---")
    st.subheader("Inference Settings")
    use_tta = st.checkbox("Enable Test-Time Augmentation (TTA)", value=True, help="Flips and multi-views for enhanced diagnostic robustness.")
    vessel_thresh = st.slider("Vessel Binarization Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">👁️ NethraAI — Explainable Retinal Diagnostic Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Diabetic Retinopathy Grading, SHAP Visual Explanations & Retinal Vessel Morphometry</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Image Loading & Preview
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload retinal fundus photograph (.png, .jpg, .jpeg, .tif)",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

input_image_path = None
temp_file_created = False

if uploaded_file is not None:
    temp_dir = tempfile.mkdtemp()
    input_image_path = os.path.join(temp_dir, uploaded_file.name)
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    temp_file_created = True
elif sample_options[selected_sample] is not None:
    candidate = sample_options[selected_sample]
    if os.path.exists(candidate):
        input_image_path = candidate
    else:
        st.warning(f"Sample file not found at: {candidate}")

if input_image_path is not None:
    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(input_image_path, caption="Active Retinal Fundus Input", use_container_width=True)
    with col_info:
        img_bgr = cv2.imread(input_image_path)
        if img_bgr is not None:
            h, w = img_bgr.shape[:2]
            st.markdown(f"**Dimensions:** `{w} x {h} px`")
            st.markdown(f"**Color Channels:** `3 (RGB)`")
            st.markdown(f"**Status:** Ready for AI Screening")

        run_diag = st.button("🚀 Run Full Diagnostic & SHAP Analysis", type="primary", use_container_width=True)

    # -------------------------------------------------------------------------
    # Analysis & Results
    # -------------------------------------------------------------------------
    if run_diag and models_ready:
        tab_dr, tab_vessels, tab_report = st.tabs([
            "🩺 DR Classification & SHAP XAI",
            "🩸 Retinal Vessel Segmentation",
            "📋 Clinical Summary Report"
        ])

        # Preprocess input image
        with st.spinner("Processing image and generating SHAP visual explanations..."):
            img_rgb_ben = preprocess_single_image(input_image_path, target_size=Config.IMG_SIZE, apply_ben_graham=True)

            # Generate SHAP explanation panel
            unique_name = f"streamlit_shap_{uuid.uuid4().hex[:8]}.png"
            panel_path = generate_confidence_panel(model, input_image_path, output_filename=unique_name)

            # Direct probability computation
            wrapper = SHAPPredictWrapper(model)
            _, probs = compute_image_shap(wrapper, img_rgb_ben)

            pred_class = int(probs.argmax())
            confidence = float(probs[pred_class])
            meta = SEVERITY_META[pred_class]

            # Vessel segmentation
            segmentor = get_vessel_segmentor()
            _, binary_mask, vessel_overlay, v_metrics = segmentor.segment_vessels(img_bgr, threshold=vessel_thresh)

        # ---------------------------------------------------------------------
        # Tab 1: DR Diagnostic & SHAP
        # ---------------------------------------------------------------------
        with tab_dr:
            st.markdown(f"""
            <div class="badge-card" style="background-color: {meta['color']};">
                <h2 style="margin: 0; color: white;">{meta['title']}</h2>
                <p style="margin: 4px 0 0 0; font-size: 1.1rem; opacity: 0.95;">Diagnostic Confidence: <b>{confidence * 100:.1f}%</b> | Risk Category: <b>{meta['urgency']}</b></p>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Calibrated Class Probabilities")
                prob_df = pd.DataFrame({
                    "Grade": [f"{i}: {Config.CLASS_LABELS[i]}" for i in range(Config.NUM_CLASSES)],
                    "Probability": [float(p) for p in probs]
                })
                st.bar_chart(prob_df.set_index("Grade"), use_container_width=True)

            with c2:
                st.subheader("Clinical Clinical Action Plan")
                st.info(f"**Recommendation:**\n\n{meta['recommendation']}")
                st.markdown(f"**ICDR Severity Standard:** Grade {pred_class} / 4")
                st.markdown(f"**Model TTA Multi-view:** {'Active' if use_tta else 'Single View'}")

            st.markdown("---")
            st.subheader("Dual-Panel SHAP Explainability & Confidence Analysis")
            if os.path.exists(panel_path):
                st.image(panel_path, caption="SHAP Attribution Heatmap (Left: Feature Importance, Right: Confidence Interval)", use_container_width=True)
                st.caption(
                    "🔴 **Red pixels:** Retinal regions strongly driving the severity assessment (microaneurysms, hemorrhages, hard exudates). "
                    "🔵 **Blue pixels:** Healthy retinal background and protective features suppressing higher severity."
                )

        # ---------------------------------------------------------------------
        # Tab 2: Vessel Segmentation
        # ---------------------------------------------------------------------
        with tab_vessels:
            st.subheader("DRIVE-Calibrated Retinal Blood Vessel Architecture")
            vcol1, vcol2 = st.columns(2)
            with vcol1:
                st.image(vessel_overlay[:, :, ::-1], caption="Fluorescent Cyan Vessel Overlay", use_container_width=True)
            with vcol2:
                st.image(binary_mask, caption="Binary Vessel Mask", use_container_width=True)

            st.markdown("### Vascular Morphometry Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("Vascular Density", f"{v_metrics.get('vessel_density_pct', 0.0):.2f}%")
            m2.metric("Mean Caliber Status", v_metrics.get('caliber_status', 'Normal'))
            m3.metric("Neovascularization Flag", "Detected" if v_metrics.get('neovascularization_detected') else "Clear")

            if v_metrics.get('neovascularization_detected'):
                st.warning("⚠️ Elevated peripheral vessel density detected. Screen for proliferative neovascular vessels or tortuosity.")
            else:
                st.success("✅ Retinal vascular architecture is within normal calibration limits.")

        # ---------------------------------------------------------------------
        # Tab 3: Clinical Summary Report
        # ---------------------------------------------------------------------
        with tab_report:
            st.subheader("Ophthalmology Screening Report")
            summary_data = {
                "Clinical Parameter": [
                    "Assessment",
                    "Severity Class",
                    "Diagnostic Confidence",
                    "Vessel Density",
                    "Neovascularization Status",
                    "Action Required",
                    "Model Backbone",
                    "Device"
                ],
                "Value": [
                    Config.CLASS_LABELS[pred_class],
                    f"Grade {pred_class} of 4",
                    f"{confidence * 100:.2f}%",
                    f"{v_metrics.get('vessel_density_pct', 0.0):.2f}%",
                    "Positive / High Risk" if v_metrics.get('neovascularization_detected') else "Negative / Normal",
                    meta['recommendation'],
                    "Swin Transformer Tiny (FP16)",
                    Config.DEVICE.upper()
                ]
            }
            st.table(pd.DataFrame(summary_data))
            st.caption("Generated by NethraAI Autonomous Screening System.")

else:
    st.info("👆 Please upload a retinal fundus image or choose one of the quick test samples from the sidebar to begin.")
