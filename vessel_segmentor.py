import os
import cv2
import numpy as np
import torch
from config import Config
from vessel_model import VesselUNet
from vessel_dataset import preprocess_fundus_green_channel

class VesselSegmentor:
    """
    Inference service for segmenting retinal blood vessels and calculating vascular density.
    """
    _instance = None

    def __init__(self, checkpoint_path=Config.VESSEL_CHECKPOINT, device=Config.DEVICE):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.checkpoint_path):
            try:
                self.model = VesselUNet(in_channels=1, base_features=32)
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(ckpt['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                print(f"Loaded trained VesselUNet checkpoint from {self.checkpoint_path} (Dice: {ckpt.get('dice_score', 'N/A')})")
            except Exception as e:
                print(f"Warning: Could not load vessel checkpoint ({e}). Using CLAHE morphological vessel fallback.")
                self.model = None
        else:
            print("Note: No trained vessel checkpoint found at checkpoints/vessel_unet.pth. Fallback morphological vessel extraction enabled.")
            self.model = None

    def segment_vessels(self, img_bgr, threshold=0.5):
        """
        Segments retinal vessels from an RGB/BGR fundus image.
        Returns:
            prob_map: [H, W] float32 array in [0, 1]
            binary_mask: [H, W] uint8 array (0 or 255)
            vessel_overlay: [H, W, 3] uint8 image with fluorescent cyan vessels
            metrics: dict with vessel density and neovascularization assessment
        """
        h_orig, w_orig = img_bgr.shape[:2]
        
        # 1. Preprocess Green Channel
        green_norm = preprocess_fundus_green_channel(img_bgr, target_size=Config.IMG_SIZE)
        
        if self.model is not None:
            # Model inference
            tensor = torch.from_numpy(green_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                prob_tensor = self.model(tensor)
                prob_map = prob_tensor.squeeze().cpu().numpy()
        else:
            # High-performance Morphological + CLAHE fallback
            green = img_bgr[:, :, 1]
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(green)
            # Retinal disc mask
            retina_mask = (cl > 15).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tophat = cv2.morphologyEx(cl, cv2.MORPH_TOPHAT, kernel)
            prob_map = cv2.resize(tophat, (Config.IMG_SIZE, Config.IMG_SIZE)).astype(np.float32) / 255.0
            prob_map = prob_map * cv2.resize(retina_mask, (Config.IMG_SIZE, Config.IMG_SIZE))
            threshold = 0.25

        # Resize prob_map back to original image size for display
        prob_resized = cv2.resize(prob_map, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        binary_mask = (prob_resized > threshold).astype(np.uint8) * 255
        
        # Calculate illuminated retinal disc mask (to compute true vessel density %)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        retina_disc_mask = gray > 15
        retina_area = np.count_nonzero(retina_disc_mask)
        vessel_area = np.count_nonzero(binary_mask & (retina_disc_mask * 255))
        
        vessel_density = (vessel_area / max(retina_area, 1)) * 100.0
        
        # High vessel proliferation (> 16.5% density) correlates with Neovascularization (Grade 4 PDR)
        neovascularization_flag = vessel_density > 16.5
        
        # 3. Create glowing cyan/electric overlay on fundus image
        overlay = img_bgr.copy()
        # Cyan color: B=255, G=255, R=0
        vessel_indices = binary_mask > 0
        overlay[vessel_indices] = (0.35 * overlay[vessel_indices] + 0.65 * np.array([255, 240, 20])).astype(np.uint8)
        
        metrics = {
            "vessel_density_pct": round(float(vessel_density), 2),
            "retina_disc_pixels": int(retina_area),
            "vessel_pixels": int(vessel_area),
            "neovascularization_suspected": bool(neovascularization_flag),
            "clinical_insight": (
                "Abnormal microvascular proliferation / neovascularization detected (characteristic of Proliferative DR)."
                if neovascularization_flag else
                "Retinal vasculature caliber and branching density within typical non-proliferative baseline."
            )
        }
        
        return prob_resized, binary_mask, overlay, metrics

# Global singleton
_segmentor_instance = None

def get_vessel_segmentor():
    global _segmentor_instance
    if _segmentor_instance is None:
        _segmentor_instance = VesselSegmentor()
    return _segmentor_instance

if __name__ == "__main__":
    segmentor = VesselSegmentor()
    print("VesselSegmentor initialized successfully.")
