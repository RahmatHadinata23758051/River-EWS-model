"""
Step 1.5 — Video Evaluation
===========================
Mengevaluasi model pada video amatir banjir.
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
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

def process_video(model, device, transform, img_size, video_path, output_path, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Gagal membuka video: {video_path.name}")
        return

    # Ambil properti video asli
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames and max_frames < total_frames:
        total_frames = max_frames

    # Konfigurasi VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"\nMemproses video: {video_path.name}")
    print(f"Resolusi Asli: {width}x{height} | Memproses: {total_frames} frames")
    
    # Gunakan amp autocast jika memungkinkan
    use_amp = (device.type == "cuda")

    pbar = tqdm(total=total_frames, desc="Processing")
    frame_count = 0
    while True:
        if max_frames and frame_count >= max_frames:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Konversi BGR (OpenCV) ke RGB (Model)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Transformasi frame
        augmented = transform(image=frame_rgb)
        input_tensor = augmented["image"].unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(input_tensor)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                pred_mask = (prob > 0.5).astype(np.uint8)

        # Resize mask kembali ke resolusi asli video menggunakan cv2 (lebih cepat dari PIL)
        pred_mask_resized = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # Post-processing: Morphological Closing untuk menutupi lubang-lubang kecil pada prediksi air
        kernel = np.ones((9, 9), np.uint8)
        pred_mask_cleaned = cv2.morphologyEx(pred_mask_resized, cv2.MORPH_CLOSE, kernel)

        # Buat Overlay (warna biru transparan pada area air)
        overlay = frame.copy()
        
        # Array NumPy masking langsung untuk mempercepat proses (BGR)
        water_area = pred_mask_cleaned == 1
        overlay[water_area, 0] = np.clip(overlay[water_area, 0] + 100, 0, 255) # Tambah Biru (B)
        overlay[water_area, 1] = overlay[water_area, 1] * 0.7 # Kurangi Hijau (G)
        overlay[water_area, 2] = overlay[water_area, 2] * 0.7 # Kurangi Merah (R)

        # Tambahkan teks HUD sederhana
        cv2.putText(overlay, "River EWS | DeepLabV3+ (ResNet-34)", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, f"GPU Trained (Colab)", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out.write(overlay)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print(f"✅ Video berhasil disimpan di: {output_path.name}")


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = BASE_DIR / "models"
    MODEL_PATH = MODELS_DIR / "best_seg_model.pth"
    
    # Path video testing dari phase 1 versi lama
    TEST_VIDEOS_DIR = BASE_DIR.parent.parent / "CV" / "test-video"
    OUTPUT_DIR = BASE_DIR / "eval_results"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("\n" + "="*50)
    print(" STEP 1.5 — EVALUASI VIDEO")
    print("="*50)
    
    if not MODEL_PATH.exists():
        print(f"❌ Model tidak ditemukan di {MODEL_PATH}")
        return

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = get_eval_transforms(img_size)

    # Uji coba pada video lain (resolusi 1080p)
    video_name = "12747554_1920_1080_24fps.mp4"
    video_path = TEST_VIDEOS_DIR / video_name
    output_path = OUTPUT_DIR / "out_vid2.mp4"

    if video_path.exists():
        # Batasi ke 300 frame (sekitar 12 detik) agar Anda tidak perlu menunggu setengah jam di CPU
        process_video(model, device, transform, img_size, video_path, output_path, max_frames=300)
    else:
        print(f"Video {video_name} tidak ditemukan di {TEST_VIDEOS_DIR}")

if __name__ == "__main__":
    main()
