import os
import cv2
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config

def preprocess_fundus_green_channel(img_bgr, target_size=Config.IMG_SIZE):
    """
    Extracts the green channel (peak vessel absorption) and applies CLAHE.
    """
    # Green channel has index 1 in BGR
    green = img_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    green_enhanced = clahe.apply(green)
    
    if target_size is not None:
        green_enhanced = cv2.resize(green_enhanced, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
    # Scale to [0, 1] float32
    norm = green_enhanced.astype(np.float32) / 255.0
    return norm

class DRIVEDataset(Dataset):
    """
    PyTorch Dataset for DRIVE Retinal Vessel Extraction.
    """
    def __init__(self, drive_root=Config.DRIVE_DIR, split="training", target_size=Config.IMG_SIZE, augment=True):
        self.target_size = target_size
        self.augment = augment and (split == "training")
        
        split_dir = os.path.join(drive_root, split)
        img_dir = os.path.join(split_dir, "images")
        
        self.image_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")) + glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
        
        # In DRIVE, masks are in 1st_manual
        manual_dir = os.path.join(split_dir, "1st_manual")
        has_manual = os.path.exists(manual_dir)
        
        self.pairs = []
        for img_p in self.image_paths:
            base = os.path.basename(img_p)
            file_id = base.split("_")[0]
            
            mask_p = None
            if has_manual:
                candidates = [
                    os.path.join(manual_dir, f"{file_id}_manual1.gif"),
                    os.path.join(manual_dir, f"{file_id}_manual1.png"),
                    os.path.join(manual_dir, f"{file_id}_manual1.tif")
                ]
                for c in candidates:
                    if os.path.exists(c):
                        mask_p = c
                        break
            self.pairs.append((img_p, mask_p))
            
        print(f"DRIVEDataset [{split}]: Loaded {len(self.pairs)} image-mask pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"Failed to read image at {img_path}")
            
        img_proc = preprocess_fundus_green_channel(img_bgr, target_size=self.target_size)
        
        if mask_path is not None and os.path.exists(mask_path):
            mask_pil = Image.open(mask_path).convert('L')
            mask_arr = np.array(mask_pil)
            if self.target_size is not None:
                mask_arr = cv2.resize(mask_arr, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_arr > 127).astype(np.float32)
        else:
            mask_bin = np.zeros((self.target_size, self.target_size), dtype=np.float32)
            
        # Data Augmentations
        if self.augment:
            if np.random.rand() > 0.5:
                img_proc = np.fliplr(img_proc).copy()
                mask_bin = np.fliplr(mask_bin).copy()
            if np.random.rand() > 0.5:
                img_proc = np.flipud(img_proc).copy()
                mask_bin = np.flipud(mask_bin).copy()
            if np.random.rand() > 0.5:
                k = np.random.choice([1, 2, 3])
                img_proc = np.rot90(img_proc, k).copy()
                mask_bin = np.rot90(mask_bin, k).copy()
                
        img_tensor = torch.from_numpy(img_proc).unsqueeze(0).float()   # [1, H, W]
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0).float()  # [1, H, W]
        
        return img_tensor, mask_tensor, os.path.basename(img_path)

def get_drive_loaders(batch_size=4, target_size=Config.IMG_SIZE):
    train_set = DRIVEDataset(split="training", target_size=target_size, augment=True)
    # Split train into train/val if small
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader
