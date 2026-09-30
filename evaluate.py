import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from config import Config
from dataset import prepare_dataloaders
from model import build_model

def apply_tta_inference(model, image_tensor):
    """
    Test-Time Augmentation (TTA):
    Calculates average softmax prediction probabilities across 3 views:
    1. Original image
    2. Horizontally flipped image
    3. Vertically flipped image
    """
    views = [
        image_tensor,
        torch.flip(image_tensor, dims=[-1]),  # horizontal flip
        torch.flip(image_tensor, dims=[-2])   # vertical flip
    ]
    
    probs_list = []
    for view in views:
        logits = model(view)
        probs = F.softmax(logits, dim=1)
        probs_list.append(probs)
        
    avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
    return avg_probs

def plot_confusion_matrix(cm, class_names, output_path=os.path.join(Config.OUTPUT_DIR, "confusion_matrix.png")):
    """
    Plots a visually rich confusion matrix heatmap and saves it to output_path.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("APTOS DR Severity Confusion Matrix (Validation Set)", fontsize=13, fontweight='bold')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)
    
    # Annotate matrix cells with numeric values and percentages
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, f"{val}",
                     horizontalalignment="center",
                     color="white" if val > thresh else "black",
                     fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.ylabel('True Diagnosis Class', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Diagnosis Class', fontsize=11, fontweight='bold')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {output_path}")

def evaluate_model(use_tta=True):
    Config.create_dirs()
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}. Please run train.py first.")

    print("=" * 60)
    print("Evaluating Best Model Checkpoint")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test-Time Augmentation (TTA): {use_tta}")
    print("=" * 60)

    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
    cfg = checkpoint.get('config', {})
    model_name = cfg.get('model_name', Config.MODEL_NAME)
    num_classes = cfg.get('num_classes', Config.NUM_CLASSES)
    use_regression = cfg.get('use_regression', getattr(Config, 'USE_REGRESSION', False))
    img_size = cfg.get('img_size', Config.IMG_SIZE)
    
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False, img_size=img_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()

    # 2. Get Validation DataLoader
    _, val_loader, _, _, val_df = prepare_dataloaders()

    all_preds = []
    all_targets = []

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
                from train import classify_predictions
                preds = classify_predictions(raw_out)
            else:
                if use_tta:
                    probs = apply_tta_inference(model, images)
                else:
                    logits = model(images)
                    probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 3. Calculate QWK Score
    qwk_score = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
    
    # 4. Generate Classification Report
    class_names = [f"Class {i}: {Config.CLASS_LABELS[i]}" for i in range(Config.NUM_CLASSES)]
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)

    # 5. Generate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(Config.NUM_CLASSES))
    plot_confusion_matrix(cm, class_names)

    # 6. Analyze Adjacent Class Confusion
    adjacent_errors = 0
    total_errors = np.sum(all_preds != all_targets)
    for i in range(len(all_preds)):
        if all_preds[i] != all_targets[i]:
            if abs(all_preds[i] - all_targets[i]) == 1:
                adjacent_errors += 1

    adjacent_ratio = (adjacent_errors / total_errors * 100) if total_errors > 0 else 0.0

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS REPORT")
    print("=" * 60)
    print(f"Quadratic Weighted Kappa (QWK) Score: {qwk_score:.4f}")
    print("\nPer-Class Classification Report:")
    print(report)
    print("\nError Breakdown:")
    print(f"Total Misclassifications: {total_errors}")
    print(f"Adjacent Class Misclassifications (e.g. Class 1 vs 2): {adjacent_errors} ({adjacent_ratio:.1f}% of errors)")
    print("Note: In diabetic retinopathy grading, adjacent class misclassification is common due to subjective clinical boundaries.")
    print("=" * 60)

    # 7. Generate ROC, AUC & Precision-Recall (PR) Curves
    try:
        from plot_curves import generate_all_curves
        generate_all_curves(use_tta=use_tta)
    except Exception as e:
        print(f"Warning: Could not generate ROC/PR curves ({e})")

    return qwk_score, report, cm

if __name__ == "__main__":
    evaluate_model()
