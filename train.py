import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import Config
from preprocess import process_dataset
from dataset import prepare_dataloaders
from model import build_model, freeze_backbone, unfreeze_backbone

def calculate_qwk(y_true, y_pred):
    """
    Computes Quadratic Weighted Kappa (QWK) metric between ground truth and predictions.
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def plot_training_curves(history, output_path=os.path.join(Config.OUTPUT_DIR, "training_curves.png")):
    """
    Plots and saves loss and QWK metric curves over training epochs.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-s', label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot QWK Score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_qwk'], 'g-^', label='Val QWK')
    plt.plot(epochs, history['val_acc'], 'm-d', label='Val Accuracy')
    plt.title('Validation QWK & Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {output_path}")

def classify_predictions(raw_preds, thresholds=[0.5, 1.5, 2.5, 3.5]):
    """
    Maps continuous regression outputs into discrete DR severity classes [0, 1, 2, 3, 4].
    """
    raw_preds = np.asarray(raw_preds).ravel()
    preds = np.zeros(len(raw_preds), dtype=int)
    for i, p in enumerate(raw_preds):
        if p < thresholds[0]:
            preds[i] = 0
        elif p < thresholds[1]:
            preds[i] = 1
        elif p < thresholds[2]:
            preds[i] = 2
        elif p < thresholds[3]:
            preds[i] = 3
        else:
            preds[i] = 4
    return preds

def train_model():
    Config.create_dirs()
    print("=" * 60)
    print("Starting APTOS 2019 Diabetic Retinopathy Model Training")
    use_regression = getattr(Config, 'USE_REGRESSION', False)
    mode_str = "MSE Ordinal Regression" if use_regression else "CrossEntropy Classification"
    print(f"Device: {Config.DEVICE} | Backbone: {Config.MODEL_NAME} | Mode: {mode_str}")
    print(f"Resolution: {Config.IMG_SIZE}x{Config.IMG_SIZE} | Batch Size: {Config.BATCH_SIZE}")
    print("=" * 60)

    # 1. Ensure dataset exists
    if not os.path.exists(Config.TRAIN_CSV) or not os.path.exists(Config.TRAIN_IMAGES_DIR):
        raise FileNotFoundError(
            f"APTOS 2019 dataset missing! Please place 'train.csv' at '{Config.TRAIN_CSV}' "
            f"and image files in '{Config.TRAIN_IMAGES_DIR}'."
        )

    # 2. Preprocess images into data/processed
    process_dataset(Config.TRAIN_IMAGES_DIR, Config.PROCESSED_DIR, target_size=Config.IMG_SIZE)

    # 3. DataLoaders & Class Weights
    train_loader, val_loader, class_weights, train_df, val_df = prepare_dataloaders()
    class_weights = class_weights.to(Config.DEVICE)

    # 4. Build Model & Loss
    num_model_classes = 1 if use_regression else Config.NUM_CLASSES
    model = build_model(Config.MODEL_NAME, num_classes=num_model_classes, pretrained=Config.PRETRAINED)
    model = model.to(Config.DEVICE)

    if use_regression:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    device_type = getattr(Config.DEVICE, 'type', str(Config.DEVICE))
    use_amp = Config.USE_AMP and (device_type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 5. Freeze backbone for first N epochs
    if Config.FREEZE_BACKBONE_EPOCHS > 0:
        freeze_backbone(model)

    best_val_qwk = -1.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_qwk': [], 'val_acc': []}

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(1, Config.EPOCHS + 1):
        # Unfreeze backbone after FREEZE_BACKBONE_EPOCHS
        if epoch == Config.FREEZE_BACKBONE_EPOCHS + 1:
            unfreeze_backbone(model)
            # Re-initialize optimizer for all unfrozen parameters
            optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        optimizer.zero_grad()

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{Config.EPOCHS:02d}] Train", leave=False)
        for step, (images, labels) in enumerate(train_bar):
            images = images.to(Config.DEVICE)
            if use_regression:
                targets = labels.to(Config.DEVICE).float().unsqueeze(1)
            else:
                targets = labels.to(Config.DEVICE)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss = loss / Config.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % Config.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            step_loss = loss.item() * Config.GRAD_ACCUM_STEPS
            running_train_loss += step_loss * images.size(0)
            train_bar.set_postfix(loss=f"{step_loss:.4f}")

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        val_preds = []
        val_targets = []

        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch:02d}/{Config.EPOCHS:02d}] Val  ", leave=False)
        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(Config.DEVICE)
                if use_regression:
                    targets = labels.to(Config.DEVICE).float().unsqueeze(1)
                else:
                    targets = labels.to(Config.DEVICE)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                running_val_loss += loss.item() * images.size(0)
                if use_regression:
                    raw_out = outputs.squeeze(1).cpu().numpy()
                    val_preds.extend(raw_out)
                else:
                    preds = torch.argmax(outputs, dim=1)
                    val_preds.extend(preds.cpu().numpy())

                val_targets.extend(labels.cpu().numpy())
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        if use_regression:
            discrete_preds = classify_predictions(val_preds)
        else:
            discrete_preds = val_preds

        val_acc = accuracy_score(val_targets, discrete_preds)
        val_qwk = calculate_qwk(val_targets, discrete_preds)

        scheduler.step()
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{Config.EPOCHS:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val QWK: {val_qwk:.4f}")

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_qwk'].append(val_qwk)
        history['val_acc'].append(val_acc)

        # Save Best Checkpoint
        if val_qwk > best_val_qwk:
            print(f"  -> QWK improved from {best_val_qwk:.4f} to {val_qwk:.4f}. Saving checkpoint...")
            best_val_qwk = val_qwk
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_qwk': best_val_qwk,
                'config': {
                    'model_name': Config.MODEL_NAME,
                    'img_size': Config.IMG_SIZE,
                    'num_classes': num_model_classes,
                    'use_regression': use_regression
                }
            }, checkpoint_path)
        else:
            patience_counter += 1
            print(f"  -> No QWK improvement for {patience_counter} epoch(s).")
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Save training curves plot
    plot_training_curves(history)

    print("=" * 60)
    print(f"Training Complete! Best Validation QWK: {best_val_qwk:.4f}")
    print(f"Best model checkpoint saved to: {checkpoint_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_model()
