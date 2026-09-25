import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from config import Config

def crop_image_from_gray(img, tol=7):
    """
    Crops black/dark borders around fundus images based on pixel thresholding.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape
        if check_shape[0] == 0 or check_shape[1] == 0:
            return img  # Return original if image is too dark / empty threshold
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            return np.stack([img1, img2, img3], axis=-1)
    return img

def apply_circular_mask(img):
    """
    Applies a circular mask to remove non-retina corners.
    """
    h, w, _ = img.shape
    center = (int(w / 2), int(h / 2))
    radius = int(min(h, w) / 2)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, (1, 1, 1), -1)
    
    masked_img = img * mask[:, :, np.newaxis]
    return masked_img

def ben_graham_preprocessing(img_rgb, img_size=224, sigmaX=10):
    """
    Ben Graham preprocessing method:
    1. Crop dark borders
    2. Resize to img_size x img_size
    3. Subtract local average color using Gaussian blur to enhance lesion contrast
    4. Apply circular mask
    """
    # 1. Crop dark borders
    cropped = crop_image_from_gray(img_rgb)
    
    # 2. Resize
    resized = cv2.resize(cropped, (img_size, img_size))
    
    # 3. Ben Graham local color average subtraction
    blurred = cv2.GaussianBlur(resized, (0, 0), sigmaX)
    enhanced = cv2.addWeighted(resized, 4, blurred, -4, 128)
    
    # 4. Circular mask
    final_img = apply_circular_mask(enhanced)
    return final_img

def preprocess_single_image(image_path, target_size=Config.IMG_SIZE, apply_ben_graham=True):
    """
    Reads an image file, applies Ben Graham preprocessing, and returns an RGB numpy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if apply_ben_graham:
        processed = ben_graham_preprocessing(img_rgb, img_size=target_size)
    else:
        cropped = crop_image_from_gray(img_rgb)
        processed = cv2.resize(cropped, (target_size, target_size))
        
    return processed

def process_dataset(raw_dir=Config.TRAIN_IMAGES_DIR, processed_dir=Config.PROCESSED_DIR, target_size=Config.IMG_SIZE):
    """
    Processes all raw fundus images and saves them as PNG files in processed_dir.
    """
    os.makedirs(processed_dir, exist_ok=True)
    if not os.path.exists(raw_dir):
        print(f"Warning: Raw images directory {raw_dir} does not exist yet.")
        return 0

    image_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    print(f"Preprocessing {len(image_files)} images from {raw_dir} -> {processed_dir}...")
    
    processed_count = 0
    for img_name in tqdm(image_files):
        src_path = os.path.join(raw_dir, img_name)
        # Ensure output filename has .png extension
        base_name = os.path.splitext(img_name)[0]
        dst_path = os.path.join(processed_dir, f"{base_name}.png")
        
        if not os.path.exists(dst_path):
            try:
                proc_img = preprocess_single_image(src_path, target_size=target_size, apply_ben_graham=True)
                cv2.imwrite(dst_path, cv2.cvtColor(proc_img, cv2.COLOR_RGB2BGR))
                processed_count += 1
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
        else:
            processed_count += 1
            
    print(f"Dataset preprocessing complete. Total images in {processed_dir}: {processed_count}")
    return processed_count

if __name__ == "__main__":
    Config.create_dirs()
    print("Testing preprocess.py module...")
    # Quick sanity check with a synthetic dummy array
    dummy_img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    processed = ben_graham_preprocessing(dummy_img, img_size=224)
    print("Processed dummy image shape:", processed.shape, "dtype:", processed.dtype)
