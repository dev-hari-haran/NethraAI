import torch
import torch.nn as nn
import timm
from config import Config

def build_model(model_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED, img_size=Config.IMG_SIZE):
    """
    Builds and returns the image classification model using timm backbones.
    Primary backbone: 'swin_tiny_patch4_window7_224'
    Fallback backbone: 'tf_efficientnet_b3_ns'
    """
    try:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size
        )
        print(f"Successfully loaded backbone: {model_name} (pretrained={pretrained}, num_classes={num_classes}, img_size={img_size})")
    except Exception as e:
        print(f"Warning: Failed to load primary backbone '{model_name}' ({e}). Falling back to '{Config.FALLBACK_MODEL_NAME}'.")
        model_name = Config.FALLBACK_MODEL_NAME
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size
        )
        print(f"Loaded fallback backbone: {model_name}")

    return model

def freeze_backbone(model):
    """
    Freezes all parameters except the final classifier head.
    Useful for initial epoch warmup.
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze head parameters (handles head, fc, classifier in timm models)
    if hasattr(model, 'head') and hasattr(model.head, 'fc'):
        for param in model.head.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
    elif hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        # Fallback: unfreeze last layer parameters
        for param in list(model.parameters())[-2:]:
            param.requires_grad = True
            
    print("Backbone parameters frozen (Classifier head trainable).")

def unfreeze_backbone(model):
    """
    Unfreezes all model parameters for full end-to-end training.
    """
    for param in model.parameters():
        param.requires_grad = True
    print("All backbone parameters unfrozen for full fine-tuning.")

if __name__ == "__main__":
    print("Testing model.py module...")
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    test_model = build_model(pretrained=False)
    output = test_model(dummy_input)
    print("Dummy input shape:", dummy_input.shape)
    print("Model output shape:", output.shape)
