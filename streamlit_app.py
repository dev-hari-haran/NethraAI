import os
import sys
import uuid
import tempfile
import hashlib
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
from shap_report import generate_confidence_panel
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
        font-size: 2.1rem;
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
active_image_id = None

if uploaded_file is not None:
    # Hash upload bytes to detect when a new file is uploaded
    file_bytes = uploaded_file.getvalue()
    active_image_id = hashlib.md5(file_bytes).hexdigest()
    temp_dir = os.path.join(Config.BASE_DIR, "tmp")
    os.makedirs(temp_dir, exist_ok=True)
    input_image_path = os.path.join(temp_dir, f"upload_{active_image_id[:10]}.png")
    if not os.path.exists(input_image_path):
        with open(input_image_path, "wb") as f:
            f.write(file_bytes)
elif sample_options[selected_sample] is not None:
    candidate = sample_options[selected_sample]
    if os.path.exists(candidate):
        input_image_path = candidate
        active_image_id = os.path.basename(candidate)
    else:
        st.warning(f"Sample file not found at: {candidate}")

# Reset session state if the image changes
if "current_image_id" not in st.session_state or st.session_state["current_image_id"] != active_image_id:
    st.session_state["current_image_id"] = active_image_id
    st.session_state["diag_result"] = None
    st.session_state["vessel_result"] = None
    st.session_state["shap_panel_path"] = None

if input_image_path is not None:
    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(input_image_path, caption="Active Retinal Fundus Input", width=360)
    with col_info:
        img_bgr = cv2.imread(input_image_path)
        if img_bgr is not None:
            h, w = img_bgr.shape[:2]
            st.markdown(f"**Dimensions:** `{w} x {h} px`")
            st.markdown(f"**Color Channels:** `3 (RGB)`")
            st.markdown(f"**Status:** Ready for AI Screening")

        run_diag = st.button("🚀 Run Full Diagnostic Screening", type="primary", use_container_width=True)

    # -------------------------------------------------------------------------
    # Execution & Session State Caching
    # -------------------------------------------------------------------------
    if run_diag and models_ready:
        with st.spinner("Executing Swin Transformer diagnosis & DRIVE U-Net vessel morphometry (< 1s)..."):
            # 1. Instant DR Diagnosis via Swin-Tiny (0.2s)
            diag_res = predict_single_image(input_image_path, model=model, use_tta=use_tta)
            st.session_state["diag_result"] = diag_res

            # 2. Instant Vessel Segmentation via U-Net (0.3s)
            segmentor = get_vessel_segmentor()
            _, binary_mask, vessel_overlay, v_metrics = segmentor.segment_vessels(img_bgr, threshold=vessel_thresh)
            st.session_state["vessel_result"] = {
                "binary_mask": binary_mask,
                "vessel_overlay": vessel_overlay,
                "metrics": v_metrics
            }

    # -------------------------------------------------------------------------
    # Display Results if Available in Session State
    # -------------------------------------------------------------------------
    if st.session_state.get("diag_result") is not None and st.session_state.get("vessel_result") is not None:
        diag_res = st.session_state["diag_result"]
        vessel_res = st.session_state["vessel_result"]

        pred_class = diag_res["class"]
        confidence = diag_res["confidence"]
        meta = SEVERITY_META[pred_class]
        probs = [diag_res["probabilities"][i] for i in range(Config.NUM_CLASSES)]

        tab_dr, tab_vessels, tab_report = st.tabs([
            "🩺 DR Classification & SHAP XAI",
            "🩸 Retinal Vessel Segmentation",
            "📋 Clinical Summary Report"
        ])

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
                st.subheader("Clinical Action Plan")
                st.info(f"**Recommendation:**\n\n{meta['recommendation']}")
                st.markdown(f"**ICDR Severity Standard:** Grade {pred_class} / 4")
                st.markdown(f"**Model TTA Multi-view:** {'Active' if use_tta else 'Single View'}")

            st.markdown("---")
            st.subheader("Dual-Panel SHAP Explainability & Confidence Analysis")

            # Check if SHAP was already computed for this session
            panel_path = st.session_state.get("shap_panel_path")

            if panel_path is None or not os.path.exists(panel_path):
                st.caption("SHAP generates pixel-level attribution to visualize microaneurysms, hemorrhages, and exudates.")
                col_btn, _ = st.columns([1, 2])
                with col_btn:
                    gen_shap = st.button("🔬 Generate SHAP Heatmap (~10s)", type="secondary", use_container_width=True)

                if gen_shap:
                    with st.spinner("Computing SHAP saliency attribution (Optimized 50 evaluations)..."):
                        unique_name = f"streamlit_shap_{uuid.uuid4().hex[:8]}.png"
                        # Lightweight 50 evaluations with use_tta=False for cloud safety (runs in ~10s without OOM)
                        panel_path = generate_confidence_panel(
                            model,
                            input_image_path,
                            output_filename=unique_name,
                            max_evals=50,
                            use_tta=False
                        )
                        st.session_state["shap_panel_path"] = panel_path
                        st.rerun()
            else:
                st.image(panel_path, caption="SHAP Attribution Heatmap (Left: Feature Importance, Right: Confidence Interval)", width=900)
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
                st.image(vessel_res["vessel_overlay"][:, :, ::-1], caption="Fluorescent Cyan Vessel Overlay", width=480)
            with vcol2:
                st.image(vessel_res["binary_mask"], caption="Binary Vessel Mask", width=480)

            st.markdown("### Vascular Morphometry Metrics")
            v_metrics = vessel_res["metrics"]
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
            v_metrics = vessel_res["metrics"]
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
