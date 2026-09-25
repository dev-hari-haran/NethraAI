import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config
from preprocess import preprocess_single_image

def get_transforms(img_size=Config.IMG_SIZE):
    """
    Returns albumentations transform pipelines for train and validation sets.
    """
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=180, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return train_transform, val_transform

class APTOSDataset(Dataset):
    """
    PyTorch Dataset for APTOS 2019 Blindness Detection.
    Loads processed images from disk or preprocesses on the fly if missing.
    """
    def __init__(self, df, raw_dir=Config.TRAIN_IMAGES_DIR, processed_dir=Config.PROCESSED_DIR, transform=None):
        self.df = df.reset_index(drop=True)
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['id_code']).strip()
        label = int(row['diagnosis'])

        base_id = os.path.splitext(img_id)[0]

        # Check for cached processed PNG image first
        processed_path = os.path.join(self.processed_dir, f"{base_id}.png")
        
        if os.path.exists(processed_path):
            img = cv2.imread(processed_path)
            if img is None:
                raise ValueError(f"Failed to read image at {processed_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Fallback to loading raw image and applying preprocessing
            raw_path = None
            for candidate_name in [img_id, f"{base_id}.png", f"{base_id}.jpg", f"{base_id}.jpeg", f"{base_id}.PNG", f"{base_id}.JPG"]:
                candidate = os.path.join(self.raw_dir, candidate_name)
                if os.path.exists(candidate):
                    raw_path = candidate
                    break

            if raw_path is None:
                # Check IDRiD directories as fallback
                idrid_id = base_id.replace("idrid_train_", "").replace("idrid_test_", "")
                idrid_candidates = [
                    os.path.join(Config.DATA_DIR, "iDRiD", "B. Disease Grading", "1. Original Images", "a. Training Set", f"{idrid_id}.jpg"),
                    os.path.join(Config.DATA_DIR, "iDRiD", "B. Disease Grading", "1. Original Images", "b. Testing Set", f"{idrid_id}.jpg")
                ]
                for c in idrid_candidates:
                    if os.path.exists(c):
                        raw_path = c
                        break

            if raw_path is None:
                raise FileNotFoundError(f"Could not find image for ID '{img_id}' in {self.raw_dir} or IDRiD paths.")

            img = preprocess_single_image(raw_path, target_size=Config.IMG_SIZE, apply_ben_graham=True)

        if self.transform is not None:
            augmented = self.transform(image=img)
            image_tensor = augmented['image']
        else:
            # Basic default normalization & tensor conversion
            img_norm = img.astype(np.float32) / 255.0
            image_tensor = torch.tensor(img_norm).permute(2, 0, 1)

        return image_tensor, torch.tensor(label, dtype=torch.long)

def prepare_dataloaders(csv_path=None, batch_size=Config.BATCH_SIZE, val_size=Config.VAL_SIZE, seed=Config.SEED):
    """
    Loads training CSV (APTOS, IDRiD, or Combined), creates stratified train/val splits,
    builds DataLoaders, and calculates class weights.
    """
    if csv_path is None:
        if getattr(Config, 'DATASET_MODE', 'combined') == 'combined' and os.path.exists(getattr(Config, 'COMBINED_TRAIN_CSV', '')):
            csv_path = Config.COMBINED_TRAIN_CSV
            print(f"Loading Multi-Source Combined Dataset (APTOS + IDRiD): {csv_path}")
        else:
            csv_path = Config.TRAIN_CSV
            print(f"Loading Dataset: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}. Please verify dataset files.")

    df = pd.read_csv(csv_path)
    
    # Safe multi-source stratification
    if 'dataset_source' in df.columns:
        df['strat_key'] = df['diagnosis'].astype(str) + "_" + df['dataset_source'].astype(str)
        # Check min class count
        min_count = df['strat_key'].value_counts().min()
        strat_col = df['strat_key'] if min_count >= 2 else df['diagnosis']
    else:
        strat_col = df['diagnosis']

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=strat_col,
        random_state=seed
    )
    
    print(f"Data split: {len(train_df)} training samples, {len(val_df)} validation samples.")
    if 'dataset_source' in df.columns:
        print("Dataset sources in Train:", dict(train_df['dataset_source'].value_counts()))
        print("Dataset sources in Val:", dict(val_df['dataset_source'].value_counts()))

    # Calculate class weights for imbalance handling
    classes = np.unique(train_df['diagnosis'])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_df['diagnosis'].values)
    
    # Fill in any missing classes if split happens to miss one (unlikely with stratification)
    class_weights_tensor = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)
    for cls_idx, w in zip(classes, weights):
        class_weights_tensor[cls_idx] = float(w)

    train_transform, val_transform = get_transforms(Config.IMG_SIZE)

    train_dataset = APTOSDataset(train_df, transform=train_transform)
    val_dataset = APTOSDataset(val_df, transform=val_transform)

    is_cuda = getattr(Config.DEVICE, 'type', str(Config.DEVICE)) == 'cuda'

    # Calculate sample weights for WeightedRandomSampler if enabled
    if getattr(Config, 'BALANCED_SAMPLING', False):
        from torch.utils.data import WeightedRandomSampler
        class_counts = train_df['diagnosis'].value_counts().to_dict()
        class_sample_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = train_df['diagnosis'].map(class_sample_weights).values
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle_train = False
    else:
        sampler = None
        shuffle_train = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=is_cuda
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=is_cuda
    )

    return train_loader, val_loader, class_weights_tensor, train_df, val_df

if __name__ == "__main__":
    Config.create_dirs()
    print("Testing dataset.py module...")
