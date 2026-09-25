import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

from config import Config
from preprocess import preprocess_single_image
from model import build_model

_global_model_cache = None

def load_inference_model(checkpoint_path=None):
    """
    Loads and caches the model checkpoint for inference.
    """
    global _global_model_cache
    if _global_model_cache is not None:
        return _global_model_cache

    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Run train.py first.")

    print(f"Loading inference model from {checkpoint_path}...")
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
    model.use_regression = use_regression

    _global_model_cache = model
    return _global_model_cache

def predict_single_image(image_input, model=None, use_tta=True):
    """
    Inference function for a single retina image file path or numpy array.
    Applies Ben Graham preprocessing, TTA inference, and returns structured predictions.
    """
    if model is None:
        model = load_inference_model()

    # 1. Preprocess input image
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        img_rgb = preprocess_single_image(image_input, target_size=Config.IMG_SIZE, apply_ben_graham=True)
    elif isinstance(image_input, np.ndarray):
        if image_input.shape[:2] != (Config.IMG_SIZE, Config.IMG_SIZE):
            img_rgb = cv2.resize(image_input, (Config.IMG_SIZE, Config.IMG_SIZE))
        else:
            img_rgb = image_input
    else:
        raise ValueError("image_input must be a file path (str) or numpy array")

    # 2. Normalize image tensor (ImageNet stats)
    img_norm = (img_rgb.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = torch.tensor(img_norm).permute(2, 0, 1).float().unsqueeze(0).to(Config.DEVICE)

    # 3. Model Prediction with optional TTA
    model.eval()
    use_regression = getattr(model, 'use_regression', getattr(Config, 'USE_REGRESSION', False))
    with torch.no_grad():
        if use_regression:
            if use_tta:
                v1 = model(tensor)
                v2 = model(torch.flip(tensor, dims=[-1]))
                v3 = model(torch.flip(tensor, dims=[-2]))
                avg_val = float(((v1 + v2 + v3) / 3.0).item())
            else:
                avg_val = float(model(tensor).item())

            from train import classify_predictions
            predicted_class = int(classify_predictions([avg_val])[0])
            dist = abs(avg_val - predicted_class)
            confidence = max(0.0, 1.0 - min(1.0, dist))
            prob_dict = {i: max(0.0, 1.0 - abs(avg_val - i)) for i in range(5)}
            total_p = sum(prob_dict.values()) + 1e-6
            prob_dict = {k: float(v / total_p) for k, v in prob_dict.items()}
        else:
            if use_tta:
                views = [
                    tensor,
                    torch.flip(tensor, dims=[-1]),
                    torch.flip(tensor, dims=[-2])
                ]
                probs_list = [F.softmax(model(v), dim=1) for v in views]
                avg_probs = torch.stack(probs_list, dim=0).mean(dim=0).squeeze(0)
            else:
                logits = model(tensor)
                avg_probs = F.softmax(logits, dim=1).squeeze(0)

            probs_np = avg_probs.cpu().numpy()
            predicted_class = int(np.argmax(probs_np))
            confidence = float(probs_np[predicted_class])

            prob_dict = {
                cls_idx: float(probs_np[cls_idx])
                for cls_idx in range(Config.NUM_CLASSES)
            }

    result = {
        "class": predicted_class,
        "label": Config.CLASS_LABELS[predicted_class],
        "confidence": confidence,
        "probabilities": prob_dict
    }

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APTOS Diabetic Retinopathy Inference")
    parser.add_argument("--image", type=str, help="Path to input fundus image", default=None)
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation")
    args = parser.parse_args()

    Config.create_dirs()

    if args.image and os.path.exists(args.image):
        test_path = args.image
    else:
        # Fallback to checking sample images in data/train_images
        sample_imgs = [f for f in os.listdir(Config.TRAIN_IMAGES_DIR) if f.endswith(('.png', '.jpg'))] if os.path.exists(Config.TRAIN_IMAGES_DIR) else []
        if sample_imgs:
            test_path = os.path.join(Config.TRAIN_IMAGES_DIR, sample_imgs[0])
            print(f"No --image supplied. Using sample image: {test_path}")
        else:
            print("No test images found in data/train_images/. Please pass --image path/to/image.png")
            exit(0)

    result = predict_single_image(test_path, use_tta=not args.no_tta)

    print("\n" + "=" * 50)
    print("INFERENCE RESULT")
    print("=" * 50)
    print(f"Input Image: {test_path}")
    print(f"Predicted Diagnosis: Class {result['class']} — {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nClass Probabilities:")
    for cls_idx, prob in result['probabilities'].items():
        print(f"  Class {cls_idx} ({Config.CLASS_LABELS[cls_idx]}): {prob:.4f}")
    print("=" * 50)
