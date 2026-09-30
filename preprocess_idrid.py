import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config
from preprocess import crop_image_from_gray, apply_circular_mask, ben_graham_preprocessing

def preprocess_idrid_image(img_bgr, target_size=Config.IMG_SIZE, sigmaX=10):
    """
    Standardizes high-resolution IDRiD fundus image:
    1. RGB conversion
    2. Dynamic border cropping
    3. Ben Graham local color subtraction
    4. Circular masking to target_size
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    processed = ben_graham_preprocessing(img_rgb, img_size=target_size, sigmaX=sigmaX)
    return cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

def process_idrid_split(img_dir, csv_path, prefix, output_dir=Config.PROCESSED_DIR, target_size=Config.IMG_SIZE):
    """
    Preprocesses IDRiD images from a split and returns a normalized dataframe.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"IDRiD ground truth CSV missing at {csv_path}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"IDRiD images directory missing at {img_dir}")
        
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    
    name_col = [c for c in df.columns if "image" in c.lower() or "name" in c.lower()][0]
    grade_col = [c for c in df.columns if "retinopathy" in c.lower() or "grade" in c.lower() or "diagnosis" in c.lower()][0]
    dme_col = [c for c in df.columns if "macular" in c.lower() or "edema" in c.lower()]
    dme_col = dme_col[0] if dme_col else None
    
    records = []
    print(f"Processing IDRiD {prefix} ({len(df)} images)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        orig_name = str(row[name_col]).strip()
        diagnosis = int(row[grade_col])
        dme = int(row[dme_col]) if dme_col and pd.notna(row[dme_col]) else -1
        
        # Determine unique filename to avoid collision between train/test IDRiD_001
        new_id_code = f"{prefix}_{orig_name}"
        dst_filename = f"{new_id_code}.png"
        dst_path = os.path.join(output_dir, dst_filename)
        
        # Find raw image file
        raw_path = None
        for ext in [".jpg", ".jpeg", ".png", ".tif", ".JPG", ".JPEG"]:
            cand = os.path.join(img_dir, orig_name + ext)
            if os.path.exists(cand):
                raw_path = cand
                break
                
        if raw_path is None:
            print(f"Warning: Image file not found for ID '{orig_name}' in {img_dir}")
            continue
            
        if not os.path.exists(dst_path):
            img_bgr = cv2.imread(raw_path)
            if img_bgr is None:
                print(f"Warning: Failed to read image {raw_path}")
                continue
            proc_bgr = preprocess_idrid_image(img_bgr, target_size=target_size)
            cv2.imwrite(dst_path, proc_bgr)
            
        records.append({
            "id_code": new_id_code,
            "diagnosis": diagnosis,
            "dme_risk": dme,
            "dataset_source": "idrid",
            "original_id": orig_name
        })
        
    return pd.DataFrame(records)

def build_combined_datasets():
    """
    Builds preprocessed IDRiD datasets and generates:
    - Data/idrid_train.csv
    - Data/idrid_test.csv
    - Data/combined_train.csv (APTOS + IDRiD Train)
    """
    idrid_base = os.path.join(Config.DATA_DIR, "iDRiD")
    grading_base = os.path.join(idrid_base, "B. Disease Grading")
    
    # 1. Paths for Train Set
    train_img_dir = os.path.join(grading_base, "1. Original Images", "a. Training Set")
    train_csv_path = os.path.join(grading_base, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
    
    # 2. Paths for Test Set
    test_img_dir = os.path.join(grading_base, "1. Original Images", "b. Testing Set")
    test_csv_path = os.path.join(grading_base, "2. Groundtruths", "b. IDRiD_Disease Grading_Testing Labels.csv")
    
    # Process Train & Test
    df_idrid_train = process_idrid_split(train_img_dir, train_csv_path, prefix="idrid_train")
    df_idrid_test = process_idrid_split(test_img_dir, test_csv_path, prefix="idrid_test")
    
    # Save IDRiD CSVs
    df_idrid_train.to_csv(Config.IDRID_TRAIN_CSV, index=False)
    df_idrid_test.to_csv(Config.IDRID_TEST_CSV, index=False)
    print(f"\nIDRiD train CSV saved to {Config.IDRID_TRAIN_CSV} ({len(df_idrid_train)} samples)")
    print(f"IDRiD test CSV saved to {Config.IDRID_TEST_CSV} ({len(df_idrid_test)} samples)")
    
    # 3. Combine with APTOS
    if os.path.exists(Config.TRAIN_CSV):
        aptos_df = pd.read_csv(Config.TRAIN_CSV)
        aptos_df = aptos_df[['id_code', 'diagnosis']].copy()
        aptos_df['dme_risk'] = -1
        aptos_df['dataset_source'] = 'aptos'
        aptos_df['original_id'] = aptos_df['id_code']
        
        combined_df = pd.concat([aptos_df, df_idrid_train], ignore_index=True)
        combined_df.to_csv(Config.COMBINED_TRAIN_CSV, index=False)
        print(f"\nCombined dataset saved to {Config.COMBINED_TRAIN_CSV}")
        print(f"Total Combined Training Samples: {len(combined_df)} (APTOS: {len(aptos_df)}, IDRiD: {len(df_idrid_train)})")
        
        print("\n--- Per-Class Distribution (Combined Train) ---")
        for cls_idx in range(Config.NUM_CLASSES):
            count = (combined_df['diagnosis'] == cls_idx).sum()
            label = Config.CLASS_LABELS[cls_idx]
            print(f"Class {cls_idx} ({label}): {count} samples")
    else:
        print(f"Warning: APTOS train.csv not found at {Config.TRAIN_CSV}.")

if __name__ == "__main__":
    Config.create_dirs()
    build_combined_datasets()
