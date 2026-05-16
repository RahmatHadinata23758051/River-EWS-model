"""
Step 1.4 — Model Evaluation & Visualization
=========================================
Load the best saved model and run inference on the test dataset.
Generates visual comparisons (Original vs Ground Truth vs Prediction)
to qualitatively evaluate model performance.
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

def get_eval_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = BASE_DIR / "models"
    MODEL_PATH = MODELS_DIR / "best_seg_model.pth"
    
    RIWA_DIR = BASE_DIR.parent / "Segmentation" / "riwa_v2"
    TEST_IMG_DIR = RIWA_DIR / "test" / "images"
    TEST_MASK_DIR = RIWA_DIR / "test" / "masks"
    
    OUTPUT_DIR = BASE_DIR / "eval_results"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("\n" + "="*50)
    print(" STEP 1.4 — EVALUASI MODEL (Visualisasi)")
    print("="*50)
    
    if not MODEL_PATH.exists():
        print(f"❌ Model tidak ditemukan di {MODEL_PATH}")
        return

    # Load Checkpoint
    print(f"Loading checkpoint: {MODEL_PATH.name}")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    cfg = checkpoint.get("cfg", {}) if isinstance(checkpoint, dict) else {}
    img_size = cfg.get("img_size", 256)
    
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    
    if "encoder" in cfg:
        encoder = cfg["encoder"]
    elif "encoder.layer1.0.conv3.weight" in state_dict:
        encoder = "resnet50"
    elif "encoder.layer3.5.conv1.weight" in state_dict:
        encoder = "resnet34"
    else:
        encoder = "resnet18" # Default fallback
    print(f"Auto-detected encoder: {encoder}")
    
    epoch_saved = checkpoint.get('epoch') if isinstance(checkpoint, dict) else 'Unknown'
    val_iou = checkpoint.get('val_iou', 0) if isinstance(checkpoint, dict) else 0
    print(f"  Epoch saved : {epoch_saved}")
    print(f"  Val IoU     : {val_iou*100:.2f}%")
    print(f"  Image Size  : {img_size}x{img_size}")
    
    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")
    
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Get sample files
    test_imgs = sorted([f for f in TEST_IMG_DIR.iterdir() if f.suffix.lower() in [".jpg", ".png"]])
    test_masks = sorted([f for f in TEST_MASK_DIR.iterdir() if f.suffix.lower() == ".png"])
    
    # Select 10 random samples or first 10
    num_samples = min(10, len(test_imgs))
    np.random.seed(42)
    sample_indices = np.random.choice(len(test_imgs), num_samples, replace=False)
    
    transform = get_eval_transforms(img_size)
    
    print(f"\nMemproses {num_samples} gambar untuk visualisasi...")
    
    for idx in tqdm(sample_indices):
        img_path = test_imgs[idx]
        mask_path = TEST_MASK_DIR / (img_path.stem + ".png") # Assuming names match
        
        if not mask_path.exists():
            continue
            
        # Read Original
        orig_img = Image.open(img_path).convert("RGB")
        orig_img_np = np.array(orig_img)
        
        orig_mask = Image.open(mask_path).convert("L")
        orig_mask_np = np.array(orig_mask)
        # Binarize mask
        if orig_mask_np.max() > 1: # RIWA format is 0/255
            orig_mask_np = (orig_mask_np > 127).astype(np.uint8)
            
        # Transform for model
        augmented = transform(image=orig_img_np)
        input_tensor = augmented["image"].unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(input_tensor)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                pred_mask = (prob > 0.5).astype(np.uint8)
        
        # Resize inputs back for display (optional, but let's just display what the model sees)
        # To make it fair, we'll resize orig image to the img_size to match prediction
        display_img = np.array(orig_img.resize((img_size, img_size)))
        display_gt = np.array(orig_mask.resize((img_size, img_size), Image.NEAREST))
        if display_gt.max() > 1: display_gt = (display_gt > 127).astype(np.uint8)

        # Plotting
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        
        axs[0].imshow(display_img)
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        
        axs[1].imshow(display_gt, cmap="gray")
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")
        
        axs[2].imshow(pred_mask, cmap="gray")
        axs[2].set_title("Predicted Mask (Model)")
        axs[2].axis("off")
        
        # Overlay
        overlay = display_img.copy()
        # Add blue tint where predicted
        overlay[pred_mask == 1, 0] = overlay[pred_mask == 1, 0] * 0.5   # R
        overlay[pred_mask == 1, 1] = overlay[pred_mask == 1, 1] * 0.5   # G
        overlay[pred_mask == 1, 2] = np.clip(overlay[pred_mask == 1, 2] + 100, 0, 255) # B
        
        axs[3].imshow(overlay)
        axs[3].set_title("Overlay Prediction")
        axs[3].axis("off")
        
        plt.tight_layout()
        # Shorten filename to avoid Windows MAX_PATH limit (>260 chars)
        save_path = OUTPUT_DIR / f"eval_{idx}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    print(f"\n✅ Evaluasi Selesai! Hasil visualisasi disimpan di: {OUTPUT_DIR}")
    print("Silakan buka gambar-gambar tersebut untuk melihat apakah model 'paham' membedakan air sungai.")

if __name__ == "__main__":
    main()
