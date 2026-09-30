/**
 * NethraAI — Frontend Controller
 * Handles image selection, backend communication, and SHAP XAI rendering.
 */

// Dynamically target backend URL (supports localStorage override for cloud/Netlify deployments)
const SAVED_API = localStorage.getItem('nethra_api_url');
const IS_SAME_ORIGIN = window.location.protocol.startsWith('http') && (window.location.port === '8000' || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
let API_BASE = (SAVED_API || (IS_SAME_ORIGIN && window.location.port === '8000' ? '' : 'http://localhost:8000')).replace(/\/+$/, '');

// Clinical Decision Support Guidance Map
const CLINICAL_GUIDELINES = {
  0: {
    name: "No Diabetic Retinopathy",
    stage: "Normal",
    desc: "Clean fundus with no microvascular lesions or hemorrhages detected.",
    advice: "Routine annual screening recommended. Maintain optimal glycemic control and healthy blood pressure.",
    sevClass: "sev-0"
  },
  1: {
    name: "Mild Non-Proliferative DR",
    stage: "Stage 1 (Mild)",
    desc: "Isolated microaneurysms detected; early microvascular dilation.",
    advice: "Follow-up screening in 6–12 months. Advise strict glycemic control and lifestyle management.",
    sevClass: "sev-1"
  },
  2: {
    name: "Moderate Non-Proliferative DR",
    stage: "Stage 2 (Moderate)",
    desc: "Multiple microaneurysms, blot hemorrhages, and early vascular changes detected.",
    advice: "Refer to an Ophthalmologist within 4–6 weeks for dilated fundus examination and OCT evaluation.",
    sevClass: "sev-2"
  },
  3: {
    name: "Severe Non-Proliferative DR",
    stage: "Stage 3 (Severe)",
    desc: "Extensive intraretinal hemorrhages and prominent vascular abnormalities observed.",
    advice: "Urgent Ophthalmology referral within 1–2 weeks. High risk of rapid progression to proliferative DR.",
    sevClass: "sev-3"
  },
  4: {
    name: "Proliferative Diabetic Retinopathy",
    stage: "Stage 4 (Proliferative)",
    desc: "Neovascularization or vitreous hemorrhage identified. Urgent clinical intervention required.",
    advice: "Immediate referral (within 24–48 hours) to a vitreoretinal specialist for laser or anti-VEGF therapy.",
    sevClass: "sev-4"
  }
};

const CLASS_NAMES = [
  "Class 0: No DR",
  "Class 1: Mild DR",
  "Class 2: Moderate DR",
  "Class 3: Severe DR",
  "Class 4: Proliferative DR"
];

// DOM Elements
const systemStatusText = document.getElementById('system-status-text');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const dropzonePrompt = document.getElementById('dropzone-prompt');
const previewBox = document.getElementById('preview-box');
const imagePreview = document.getElementById('image-preview');
const clearBtn = document.getElementById('clear-btn');
const fileInfo = document.getElementById('file-info');
const fileNameLabel = document.getElementById('file-name-label');
const fileSizeLabel = document.getElementById('file-size-label');
const analyzeBtn = document.getElementById('analyze-btn');
const analyzeBtnText = document.getElementById('analyze-btn-text');

// State views
const standbyView = document.getElementById('standby-view');
const loadingView = document.getElementById('loading-view');
const resultsView = document.getElementById('results-view');
const loadingTitle = document.getElementById('loading-title');
const loadingDesc = document.getElementById('loading-desc');

// Result elements
const severityPill = document.getElementById('severity-pill');
const classCode = document.getElementById('class-code');
const resultDiagnosis = document.getElementById('result-diagnosis');
const resultDesc = document.getElementById('result-desc');
const confValue = document.getElementById('conf-value');
const adviceText = document.getElementById('advice-text');
const probList = document.getElementById('prob-list');
const shapImg = document.getElementById('shap-img');
const shapImgBox = document.getElementById('shap-img-box');
const zoomBtn = document.getElementById('zoom-btn');

// Modal Elements
const modal = document.getElementById('modal');
const modalOverlay = document.getElementById('modal-overlay');
const modalClose = document.getElementById('modal-close');
const modalImg = document.getElementById('modal-img');

let selectedFile = null;
let statusCycle = null;

// ============================================================================
// 1. Initial Health Check
// ============================================================================
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      systemStatusText.textContent = `Model Ready (${data.device.toUpperCase()} Accelerated)`;
      document.querySelector('.status-dot').style.backgroundColor = '#10b981';
    } else {
      systemStatusText.textContent = 'Model Offline (Click to configure)';
      document.querySelector('.status-dot').style.backgroundColor = '#ef4444';
    }
  } catch (err) {
    systemStatusText.textContent = 'Model Offline (Click to configure)';
    document.querySelector('.status-dot').style.backgroundColor = '#ef4444';
  }
}
checkHealth();

// Allow configuring backend URL by clicking status pill (useful on Netlify)
const statusPill = document.getElementById('system-status');
if (statusPill) {
  statusPill.style.cursor = 'pointer';
  statusPill.title = 'Click to configure backend API endpoint';
  statusPill.addEventListener('click', () => {
    const current = localStorage.getItem('nethra_api_url') || API_BASE;
    const input = prompt("Configure NethraAI backend API URL:\n(e.g., https://your-backend.onrender.com or http://localhost:8000)", current);
    if (input !== null) {
      const trimmed = input.trim().replace(/\/+$/, '');
      if (trimmed) {
        localStorage.setItem('nethra_api_url', trimmed);
        API_BASE = trimmed;
      } else {
        localStorage.removeItem('nethra_api_url');
        API_BASE = IS_SAME_ORIGIN && window.location.port === '8000' ? '' : 'http://localhost:8000';
      }
      systemStatusText.textContent = 'Connecting...';
      document.querySelector('.status-dot').style.backgroundColor = '#f59e0b';
      checkHealth();
    }
  });
}

// ============================================================================
// 2. File Selection & Drag-and-Drop
// ============================================================================
function handleFile(file) {
  if (!file) return;

  const validExts = ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'];
  const ext = file.name.split('.').pop().toLowerCase();
  if (!validExts.includes(ext)) {
    alert("Please select a supported fundus image format (PNG, JPG, TIFF, BMP).");
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    dropzonePrompt.classList.add('hidden');
    previewBox.classList.remove('hidden');

    fileNameLabel.textContent = file.name;
    fileSizeLabel.textContent = formatBytes(file.size);
    fileInfo.classList.remove('hidden');

    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function clearSelection() {
  selectedFile = null;
  fileInput.value = '';
  imagePreview.src = '';
  previewBox.classList.add('hidden');
  dropzonePrompt.classList.remove('hidden');
  fileInfo.classList.add('hidden');
  analyzeBtn.disabled = true;
}

// Drag & drop listeners
['dragenter', 'dragover'].forEach(ev => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach(ev => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  });
});

dropzone.addEventListener('drop', (e) => {
  const files = e.dataTransfer?.files;
  if (files && files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files.length > 0) handleFile(e.target.files[0]);
});

clearBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearSelection();
});

// ============================================================================
// 3. Quick Patient Case Loaders
// ============================================================================
document.querySelectorAll('.case-btn').forEach(btn => {
  btn.addEventListener('click', async (e) => {
    const sample = btn.getAttribute('data-sample');
    const originalText = btn.innerHTML;
    try {
      btn.textContent = 'Loading Scan...';
      let res;
      try {
        res = await fetch(`${API_BASE}/sample-image/${encodeURIComponent(sample)}`);
      } catch (e) {
        res = null;
      }
      if (!res || !res.ok) {
        res = await fetch(`samples/${encodeURIComponent(sample)}`);
      }
      if (!res.ok) throw new Error("Sample file not available.");
      const blob = await res.blob();
      const file = new File([blob], sample, { type: blob.type || 'image/png' });
      handleFile(file);
    } catch (err) {
      alert(`Could not load test case: ${err.message}`);
    } finally {
      btn.innerHTML = originalText;
    }
  });
});

// ============================================================================
// 4. Diagnostic & SHAP Analysis
// ============================================================================
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  setLoading(true);

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    // Fetch both model screening and DRIVE vasculature extraction
    const [predictRes, vesselRes] = await Promise.allSettled([
      fetch(`${API_BASE}/predict`, { method: 'POST', body: formData }),
      fetch(`${API_BASE}/api/vessel-segmentation`, { method: 'POST', body: formData })
    ]);

    if (predictRes.status !== 'fulfilled' || !predictRes.value.ok) {
      const err = predictRes.status === 'fulfilled' ? await predictRes.value.json().catch(() => ({})) : {};
      throw new Error(err.detail || "Diagnostic screening request failed.");
    }

    const data = await predictRes.value.json();
    let vesselData = null;
    if (vesselRes.status === 'fulfilled' && vesselRes.value.ok) {
      vesselData = await vesselRes.value.json().catch(() => null);
    }

    renderAssessment(data, vesselData);
  } catch (error) {
    alert(`Diagnosis Error: ${error.message}`);
    standbyView.classList.remove('hidden');
    loadingView.classList.add('hidden');
    resultsView.classList.add('hidden');
  } finally {
    setLoading(false);
  }
});

// ============================================================================
// 5. Render Assessment Results
// ============================================================================
function renderAssessment(data, vesselData) {
  const predClass = data.predicted_class;
  const conf = data.confidence;
  const probs = data.all_probs;
  const b64 = data.shap_panel_base64;

  const info = CLINICAL_GUIDELINES[predClass] || {
    name: data.predicted_label,
    stage: `Stage ${predClass}`,
    desc: "Diagnostic evaluation complete.",
    advice: "Consult an ophthalmologist for professional verification.",
    sevClass: "sev-2"
  };

  // 1. Diagnosis Banner
  severityPill.textContent = info.stage;
  severityPill.className = `severity-pill ${info.sevClass}`;
  classCode.textContent = `ICDR Class ${predClass}`;
  resultDiagnosis.textContent = info.name;
  resultDesc.textContent = info.desc;
  confValue.textContent = `${(conf * 100).toFixed(1)}%`;

  // 2. Clinical Advice
  adviceText.textContent = info.advice;

  // 3. Probabilities
  probList.innerHTML = '';
  probs.forEach((p, i) => {
    const isPred = i === predClass;
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <span class="prob-label ${isPred ? 'active' : ''}">${CLASS_NAMES[i]}</span>
      <div class="prob-track">
        <div class="prob-fill ${isPred ? 'active' : ''}" style="width: 0%"></div>
      </div>
      <span class="prob-val ${isPred ? 'active' : ''}">${(p * 100).toFixed(1)}%</span>
    `;
    probList.appendChild(row);

    setTimeout(() => {
      row.querySelector('.prob-fill').style.width = `${Math.max(1, p * 100)}%`;
    }, 40);
  });

  // 4. SHAP Artifact Image
  const imgSrc = b64.startsWith('data:') ? b64 : `data:image/png;base64,${b64}`;
  shapImg.src = imgSrc;
  modalImg.src = imgSrc;

  // 5. DRIVE Vasculature Overlay & Biomarkers
  if (vesselData && vesselData.vessel_overlay_base64) {
    const vSrc = `data:image/png;base64,${vesselData.vessel_overlay_base64}`;
    const vImg = document.getElementById('vessel-img');
    if (vImg) vImg.src = vSrc;

    const m = vesselData.metrics || {};
    const densityEl = document.getElementById('metric-density');
    const densityPill = document.getElementById('vessel-density-pill');
    const neovasEl = document.getElementById('metric-neovas');
    const statusPill = document.getElementById('vessel-status-pill');
    const descEl = document.getElementById('metric-desc');

    if (densityEl) densityEl.textContent = `${m.vessel_density_pct || 0}%`;
    if (densityPill) densityPill.textContent = `Density: ${m.vessel_density_pct || 0}%`;
    
    const isSuspected = !!m.neovascularization_suspected;
    if (neovasEl) {
      neovasEl.textContent = isSuspected ? "High (Suspected PDR)" : "Normal Non-Proliferative";
      neovasEl.style.color = isSuspected ? "var(--dr-4)" : "var(--dr-0)";
    }
    if (statusPill) {
      statusPill.textContent = isSuspected ? "Neovascularization Risk" : "Normal Caliber";
      statusPill.className = `vessel-status-pill ${isSuspected ? 'suspected' : ''}`;
    }
    if (descEl) descEl.textContent = m.clinical_insight || "Retinal vascular structure extracted via DRIVE U-Net.";
  }

  // Show results view
  standbyView.classList.add('hidden');
  loadingView.classList.add('hidden');
  resultsView.classList.remove('hidden');
}

// Wire XAI Tab Switching
const tabShapBtn = document.getElementById('tab-shap-btn');
const tabVesselBtn = document.getElementById('tab-vessel-btn');
const shapCard = document.getElementById('shap-card');
const vesselCard = document.getElementById('vessel-card');
const vesselImgBox = document.getElementById('vessel-img-box');

if (tabShapBtn && tabVesselBtn) {
  tabShapBtn.addEventListener('click', () => {
    tabShapBtn.classList.add('active');
    tabVesselBtn.classList.remove('active');
    if (shapCard) shapCard.classList.remove('hidden');
    if (vesselCard) vesselCard.classList.add('hidden');
  });
  tabVesselBtn.addEventListener('click', () => {
    tabVesselBtn.classList.add('active');
    tabShapBtn.classList.remove('active');
    if (vesselCard) vesselCard.classList.remove('hidden');
    if (shapCard) shapCard.classList.add('hidden');
  });
}

if (vesselImgBox) {
  vesselImgBox.addEventListener('click', () => {
    const vImg = document.getElementById('vessel-img');
    if (vImg && vImg.src) {
      modalImg.src = vImg.src;
      modal.classList.remove('hidden');
    }
  });
}

// ============================================================================
// 6. UI Helpers & Modal Lightbox
// ============================================================================
function setLoading(isLoading) {
  if (isLoading) {
    analyzeBtn.disabled = true;
    analyzeBtnText.textContent = "Processing Scan...";
    standbyView.classList.add('hidden');
    resultsView.classList.add('hidden');
    loadingView.classList.remove('hidden');

    const steps = [
      { t: "Preprocessing Retinal Scan...", d: "Applying Ben Graham local color normalization & circular mask..." },
      { t: "Evaluating Swin Transformer...", d: "Performing deep feature extraction with Test-Time Augmentation..." },
      { t: "Generating SHAP Attributions...", d: "Calculating Monte Carlo pixel importance masks..." }
    ];
    let idx = 0;
    loadingTitle.textContent = steps[0].t;
    loadingDesc.textContent = steps[0].d;

    statusCycle = setInterval(() => {
      idx = (idx + 1) % steps.length;
      loadingTitle.textContent = steps[idx].t;
      loadingDesc.textContent = steps[idx].d;
    }, 2800);
  } else {
    analyzeBtn.disabled = false;
    analyzeBtnText.textContent = "Run Diagnostic Screening";
    if (statusCycle) clearInterval(statusCycle);
  }
}

function openZoom() {
  if (shapImg.src) modal.classList.remove('hidden');
}
function closeZoom() {
  modal.classList.add('hidden');
}

zoomBtn.addEventListener('click', openZoom);
shapImgBox.addEventListener('click', openZoom);
modalClose.addEventListener('click', closeZoom);
modalOverlay.addEventListener('click', closeZoom);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeZoom();
});

function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}
