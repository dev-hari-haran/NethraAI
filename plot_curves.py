import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from config import Config
from dataset import prepare_dataloaders
from model import build_model
from evaluate import apply_tta_inference

# Define custom harmonious color palette for 5 DR classes
CLASS_COLORS = [
    '#2b5c8f',  # Class 0: No DR (Blue)
    '#2ca02c',  # Class 1: Mild DR (Green)
    '#ff7f0e',  # Class 2: Moderate DR (Orange)
    '#d62728',  # Class 3: Severe DR (Red)
    '#9467bd'   # Class 4: Proliferative DR (Purple)
]

MICRO_COLOR = '#1f77b4'
MACRO_COLOR = '#8c564b'
BINARY_COLOR = '#e377c2'

def extract_model_predictions(use_tta=True):
    """
    Loads best model checkpoint and runs inference on validation set.
    Returns:
        all_targets: np.ndarray of shape (N,)
        probs: np.ndarray of shape (N, 5) with class probabilities
        raw_preds: np.ndarray of shape (N,) with continuous predictions (if regression mode)
        use_regression: bool
    """
    Config.create_dirs()
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please run train.py first.")

    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
    cfg = checkpoint.get('config', {})
    model_name = cfg.get('model_name', Config.MODEL_NAME)
    num_classes = cfg.get('num_classes', Config.NUM_CLASSES)
    use_regression = cfg.get('use_regression', getattr(Config, 'USE_REGRESSION', False))
    img_size = cfg.get('img_size', Config.IMG_SIZE)

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Model Backbone: {model_name} | Mode: {'Regression' if use_regression else 'Classification'}")

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False, img_size=img_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()

    _, val_loader, _, _, _ = prepare_dataloaders()

    all_targets = []
    all_raw_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(Config.DEVICE)
            if use_regression:
                if use_tta:
                    out1 = model(images)
                    out2 = model(torch.flip(images, dims=[3]))
                    raw_out = ((out1 + out2) / 2.0).squeeze(1).cpu().numpy()
                else:
                    raw_out = model(images).squeeze(1).cpu().numpy()

                all_raw_preds.extend(raw_out)
                
                # Convert continuous regression outputs to calibrated class probabilities using RBF kernel
                # distance to class indices [0, 1, 2, 3, 4]
                sigma = 0.75
                dists = - (raw_out[:, None] - np.arange(5)[None, :]) ** 2 / (2 * (sigma ** 2))
                probs_batch = F.softmax(torch.tensor(dists), dim=1).numpy()
                all_probs.extend(probs_batch)
            else:
                if use_tta:
                    probs_batch = apply_tta_inference(model, images).cpu().numpy()
                else:
                    logits = model(images)
                    probs_batch = F.softmax(logits, dim=1).cpu().numpy()

                all_probs.extend(probs_batch)
                all_raw_preds.extend(np.argmax(probs_batch, axis=1))

            all_targets.extend(labels.numpy())

    return np.array(all_targets), np.array(all_probs), np.array(all_raw_preds), use_regression

def plot_roc_curves(all_targets, probs, raw_preds, use_regression, save_path=None):
    """
    Plots per-class One-vs-Rest ROC curves, micro/macro averages, and binary DR ROC curve.
    """
    if save_path is None:
        save_path = os.path.join(Config.OUTPUT_DIR, "roc_curves.png")

    num_classes = Config.NUM_CLASSES
    y_bin = label_binarize(all_targets, classes=list(range(num_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Per-class ROC
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average ROC
    roc_auc["macro"] = roc_auc_score(y_bin, probs, average="macro")

    # Binary DR Detection ROC (Any DR [Class 1-4] vs No DR [Class 0])
    y_bin_dr = (all_targets > 0).astype(int)
    if use_regression:
        p_dr = 1.0 / (1.0 + np.exp(-(raw_preds - 0.5)))
    else:
        p_dr = 1.0 - probs[:, 0]
    fpr_dr, tpr_dr, _ = roc_curve(y_bin_dr, p_dr)
    roc_auc_dr = auc(fpr_dr, tpr_dr)

    plt.figure(figsize=(9, 7), dpi=300)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC = 0.5000)')

    # Plot Binary DR Detection
    plt.plot(fpr_dr, tpr_dr, color=BINARY_COLOR, lw=2.5, linestyle='-',
             label=f'Binary DR Detection (AUC = {roc_auc_dr:.4f})')

    # Plot Micro & Macro averages
    plt.plot(fpr["micro"], tpr["micro"], color=MICRO_COLOR, lw=2, linestyle=':',
             label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.4f})')

    # Plot Per-Class ROC curves
    for i in range(num_classes):
        cls_name = Config.CLASS_LABELS[i]
        plt.plot(fpr[i], tpr[i], color=CLASS_COLORS[i], lw=2,
                 label=f'Class {i}: {cls_name} (AUC = {roc_auc[i]:.4f})')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curves\nAPTOS 2019 Model Evaluation', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc='lower right', fontsize=9.5, framealpha=0.9, edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC Curves saved to {save_path}")

    return {
        'per_class_auc': roc_auc,
        'micro_auc': roc_auc["micro"],
        'macro_auc': roc_auc["macro"],
        'binary_dr_auc': roc_auc_dr
    }

def plot_pr_curves(all_targets, probs, raw_preds, use_regression, save_path=None):
    """
    Plots per-class One-vs-Rest Precision-Recall curves, micro average, and binary DR PR curve.
    """
    if save_path is None:
        save_path = os.path.join(Config.OUTPUT_DIR, "pr_curves.png")

    num_classes = Config.NUM_CLASSES
    y_bin = label_binarize(all_targets, classes=list(range(num_classes)))

    precision = dict()
    recall = dict()
    ap_score = dict()

    # Per-class PR
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap_score[i] = average_precision_score(y_bin[:, i], probs[:, i])

    # Micro-average PR
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_bin.ravel(), probs.ravel())
    ap_score["micro"] = average_precision_score(y_bin, probs, average="micro")

    # Macro-average PR AP score
    ap_score["macro"] = average_precision_score(y_bin, probs, average="macro")

    # Binary DR Detection PR (Any DR vs No DR)
    y_bin_dr = (all_targets > 0).astype(int)
    if use_regression:
        p_dr = 1.0 / (1.0 + np.exp(-(raw_preds - 0.5)))
    else:
        p_dr = 1.0 - probs[:, 0]
    precision_dr, recall_dr, _ = precision_recall_curve(y_bin_dr, p_dr)
    ap_score_dr = average_precision_score(y_bin_dr, p_dr)

    plt.figure(figsize=(9, 7), dpi=300)

    # Plot Binary DR Detection
    plt.plot(recall_dr, precision_dr, color=BINARY_COLOR, lw=2.5, linestyle='-',
             label=f'Binary DR Detection (AP = {ap_score_dr:.4f})')

    # Plot Micro-average PR
    plt.plot(recall["micro"], precision["micro"], color=MICRO_COLOR, lw=2, linestyle=':',
             label=f'Micro-average PR (AP = {ap_score["micro"]:.4f})')

    # Plot Per-Class PR curves
    for i in range(num_classes):
        cls_name = Config.CLASS_LABELS[i]
        plt.plot(recall[i], precision[i], color=CLASS_COLORS[i], lw=2,
                 label=f'Class {i}: {cls_name} (AP = {ap_score[i]:.4f})')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall (PR) Curves\nAPTOS 2019 Model Evaluation', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc='lower left', fontsize=9.5, framealpha=0.9, edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"PR Curves saved to {save_path}")

    return {
        'per_class_ap': ap_score,
        'micro_ap': ap_score["micro"],
        'macro_ap': ap_score["macro"],
        'binary_dr_ap': ap_score_dr
    }

def plot_combined_roc_pr_dashboard(all_targets, probs, raw_preds, use_regression, save_path=None):
    """
    Generates a 1x2 panel dashboard figure combining both ROC and PR curves side-by-side.
    """
    if save_path is None:
        save_path = os.path.join(Config.OUTPUT_DIR, "model_roc_pr_curves.png")

    num_classes = Config.NUM_CLASSES
    y_bin = label_binarize(all_targets, classes=list(range(num_classes)))

    # --- 1. ROC Computations ---
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    roc_auc["macro"] = roc_auc_score(y_bin, probs, average="macro")

    y_bin_dr = (all_targets > 0).astype(int)
    p_dr = 1.0 / (1.0 + np.exp(-(raw_preds - 0.5))) if use_regression else (1.0 - probs[:, 0])
    fpr_dr, tpr_dr, _ = roc_curve(y_bin_dr, p_dr)
    roc_auc_dr = auc(fpr_dr, tpr_dr)

    # --- 2. PR Computations ---
    precision, recall, ap_score = {}, {}, {}
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap_score[i] = average_precision_score(y_bin[:, i], probs[:, i])
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_bin.ravel(), probs.ravel())
    ap_score["micro"] = average_precision_score(y_bin, probs, average="micro")
    ap_score["macro"] = average_precision_score(y_bin, probs, average="macro")

    precision_dr, recall_dr, _ = precision_recall_curve(y_bin_dr, p_dr)
    ap_score_dr = average_precision_score(y_bin_dr, p_dr)

    # --- 3. Figure Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Subplot 1: ROC Curve
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (AUC = 0.50)')
    ax1.plot(fpr_dr, tpr_dr, color=BINARY_COLOR, lw=2.5, label=f'Binary DR (AUC = {roc_auc_dr:.4f})')
    ax1.plot(fpr["micro"], tpr["micro"], color=MICRO_COLOR, lw=2, linestyle=':', label=f'Micro-avg (AUC = {roc_auc["micro"]:.4f})')
    for i in range(num_classes):
        ax1.plot(fpr[i], tpr[i], color=CLASS_COLORS[i], lw=2, label=f'{Config.CLASS_LABELS[i]} ({roc_auc[i]:.4f})')
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.05])
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: PR Curve
    ax2.plot(recall_dr, precision_dr, color=BINARY_COLOR, lw=2.5, label=f'Binary DR (AP = {ap_score_dr:.4f})')
    ax2.plot(recall["micro"], precision["micro"], color=MICRO_COLOR, lw=2, linestyle=':', label=f'Micro-avg (AP = {ap_score["micro"]:.4f})')
    for i in range(num_classes):
        ax2.plot(recall[i], precision[i], color=CLASS_COLORS[i], lw=2, label=f'{Config.CLASS_LABELS[i]} ({ap_score[i]:.4f})')
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold')
    ax2.set_title('Precision-Recall (PR) Curves', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=8.5, framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    fig.suptitle('Model Evaluation Dashboard: ROC, AUC & Precision-Recall Analysis', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Combined ROC & PR Dashboard saved to {save_path}")

def generate_all_curves(use_tta=True):
    """
    Main execution function to compute predictions and generate all ROC, AUC, and PR graphs.
    """
    print("=" * 60)
    print("GENERATING MODEL ROC, AUC & PRECISION-RECALL (PR) GRAPHS")
    print("=" * 60)

    all_targets, probs, raw_preds, use_regression = extract_model_predictions(use_tta=use_tta)

    roc_metrics = plot_roc_curves(all_targets, probs, raw_preds, use_regression)
    pr_metrics = plot_pr_curves(all_targets, probs, raw_preds, use_regression)
    plot_combined_roc_pr_dashboard(all_targets, probs, raw_preds, use_regression)

    # Print Summary Performance Metrics Table
    print("\n" + "=" * 60)
    print("MODEL ROC-AUC & PR-AUC PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Class / Metric':<30} | {'ROC-AUC':<10} | {'PR-AUC (AP)':<12}")
    print("-" * 60)

    for i in range(Config.NUM_CLASSES):
        cls_label = f"Class {i}: {Config.CLASS_LABELS[i]}"
        roc_val = roc_metrics['per_class_auc'][i]
        pr_val = pr_metrics['per_class_ap'][i]
        print(f"{cls_label:<30} | {roc_val:<10.4f} | {pr_val:<12.4f}")

    print("-" * 60)
    print(f"{'Micro-Average':<30} | {roc_metrics['micro_auc']:<10.4f} | {pr_metrics['micro_ap']:<12.4f}")
    print(f"{'Macro-Average':<30} | {roc_metrics['macro_auc']:<10.4f} | {pr_metrics['macro_ap']:<12.4f}")
    print(f"{'Binary DR Detection (Class >= 1)':<30} | {roc_metrics['binary_dr_auc']:<10.4f} | {pr_metrics['binary_dr_ap']:<12.4f}")
    print("=" * 60)

if __name__ == "__main__":
    generate_all_curves()
