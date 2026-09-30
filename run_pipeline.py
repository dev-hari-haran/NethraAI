import argparse
import sys
import os

from config import Config

def main():
    parser = argparse.ArgumentParser(description="APTOS 2019 Diabetic Retinopathy Pipeline Control")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["preprocess", "train", "evaluate", "curves", "gradcam", "shap-report", "infer", "all"],
        default="train",
        help="Pipeline step to execute: preprocess | train | evaluate | curves | gradcam | shap-report | infer | all"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        default=None,
        help="Path to single image file for inference (--mode infer)"
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable Test-Time Augmentation during evaluation or inference"
    )

    args = parser.parse_args()
    Config.create_dirs()

    if args.mode == "preprocess":
        from preprocess import process_dataset
        print("Running image preprocessing...")
        process_dataset(Config.TRAIN_IMAGES_DIR, Config.PROCESSED_DIR, target_size=Config.IMG_SIZE)

    elif args.mode == "train":
        from train import train_model
        print("Starting model training...")
        train_model()

    elif args.mode == "evaluate":
        from evaluate import evaluate_model
        print("Starting model evaluation...")
        evaluate_model(use_tta=not args.no_tta)

    elif args.mode == "curves":
        from plot_curves import generate_all_curves
        print("Generating ROC, AUC and Precision-Recall (PR) Curves...")
        generate_all_curves(use_tta=not args.no_tta)

    elif args.mode == "gradcam":
        from gradcam import save_class_gradcam_examples, visualize_single_image_gradcam
        print("Generating Grad-CAM explainability maps...")
        if args.image:
            visualize_single_image_gradcam(args.image)
        else:
            save_class_gradcam_examples()

    elif args.mode == "shap-report":
        from shap_report import run_full_shap_report, visualize_single_image_shap
        print("Generating SHAP XAI Report & Faithfulness Artifacts...")
        if args.image:
            visualize_single_image_shap(args.image)
        else:
            run_full_shap_report()

    elif args.mode == "infer":
        from infer import predict_single_image
        if not args.image:
            print("Error: --image argument required for infer mode. Example: python run_pipeline.py --mode infer --image Data/train_images/sample.png")
            sys.exit(1)
        result = predict_single_image(args.image, use_tta=not args.no_tta)
        print("\n" + "=" * 50)
        print("INFERENCE RESULT")
        print("=" * 50)
        print(f"Input Image: {args.image}")
        print(f"Predicted Diagnosis: Class {result['class']} — {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nClass Probabilities:")
        for cls_idx, prob in result['probabilities'].items():
            print(f"  Class {cls_idx} ({Config.CLASS_LABELS[cls_idx]}): {prob:.4f}")
        print("=" * 50)

    elif args.mode == "all":
        from preprocess import process_dataset
        from train import train_model
        from evaluate import evaluate_model
        from gradcam import save_class_gradcam_examples

        print("Executing Full Pipeline (Preprocess -> Train -> Evaluate -> GradCAM)...")
        process_dataset(Config.TRAIN_IMAGES_DIR, Config.PROCESSED_DIR, target_size=Config.IMG_SIZE)
        train_model()
        evaluate_model(use_tta=not args.no_tta)
        save_class_gradcam_examples()
        print("\nAll pipeline steps completed successfully!")

if __name__ == "__main__":
    main()
