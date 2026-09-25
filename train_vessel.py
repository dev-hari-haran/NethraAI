import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from vessel_model import VesselUNet
from vessel_dataset import DRIVEDataset

class DiceBCELoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Soft Dice Loss for retinal vessel segmentation.
    """
    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice
        
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss, dice.item()

def train_vessel_segmentor(epochs=40, batch_size=4, lr=1e-3):
    Config.create_dirs()
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Training Retinal Vessel Segmentation Model (DRIVE Dataset)")
    print(f"Device: {Config.DEVICE}")
    print(f"Image Resolution: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {lr}")
    print("=" * 60)
    
    # 1. Dataset & DataLoader
    dataset = DRIVEDataset(drive_root=Config.DRIVE_DIR, split="training", target_size=Config.IMG_SIZE, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 2. Model, Loss, Optimizer
    model = VesselUNet(in_channels=1, base_features=32).to(Config.DEVICE)
    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_dice = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{epochs}")
        for imgs, masks, _ in pbar:
            imgs = imgs.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss, dice = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            running_dice += dice * imgs.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice:.4f}"})
            
        epoch_loss = running_loss / len(dataset)
        epoch_dice = running_dice / len(dataset)
        scheduler.step()
        
        print(f"Epoch {epoch:02d} | Avg Loss: {epoch_loss:.4f} | Avg Dice Score: {epoch_dice:.4f}")
        
        if epoch_dice > best_dice:
            best_dice = epoch_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'dice_score': best_dice,
                'target_size': Config.IMG_SIZE
            }, Config.VESSEL_CHECKPOINT)
            print(f" -> Best vessel checkpoint saved to {Config.VESSEL_CHECKPOINT} (Dice: {best_dice:.4f})")
            
    print("\n" + "=" * 60)
    print(f"Vessel Segmentation Training Completed! Best Dice: {best_dice:.4f}")
    print(f"Checkpoint location: {Config.VESSEL_CHECKPOINT}")
    print("=" * 60)

if __name__ == "__main__":
    train_vessel_segmentor()
