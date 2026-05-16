import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- KONFIGURASI PATH ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PHASE1_DIR = BASE_DIR / "phase1_segmentation"
MODEL_PATH = PHASE1_DIR / "models" / "best_seg_model.pth"
TEST_VIDEOS_DIR = BASE_DIR.parent / "CV" / "test-video"
CONFIG_PATH = BASE_DIR / "phase3_integration" / "configs" / "calibration.json"
OUTPUT_DIR = BASE_DIR / "phase2_gauge" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Status Thresholds (Contoh)
THRESHOLDS = {
    "NORMAL": {"max": 100, "color": (0, 255, 0)},     # Hijau
    "SIAGA": {"max": 200, "color": (0, 255, 255)},    # Kuning
    "WASPADA": {"max": 250, "color": (0, 165, 255)},  # Oranye
    "AWAS": {"max": 9999, "color": (0, 0, 255)}       # Merah
}

def get_status(cm):
    if cm < THRESHOLDS["NORMAL"]["max"]: return "NORMAL", THRESHOLDS["NORMAL"]["color"]
    if cm < THRESHOLDS["SIAGA"]["max"]: return "SIAGA", THRESHOLDS["SIAGA"]["color"]
    if cm < THRESHOLDS["WASPADA"]["max"]: return "WASPADA", THRESHOLDS["WASPADA"]["color"]
    return "AWAS", THRESHOLDS["AWAS"]["color"]

def get_eval_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def main():
    print("\n" + "="*50)
    print(" STEP 2.2 — PENGUKURAN TINGGI AIR (VIRTUAL GAUGE)")
    print("="*50)

    # 1. BACA KALIBRASI
    if not CONFIG_PATH.exists():
        print(f"❌ File kalibrasi tidak ditemukan: {CONFIG_PATH}")
        print("Silakan jalankan 01_calibrate_gauge.py terlebih dahulu!")
        return

    with open(CONFIG_PATH, "r") as f:
        calib = json.load(f)

    if calib.get("status") != "CALIBRATED":
        print("❌ Status kalibrasi belum selesai. Jalankan 01_calibrate_gauge.py!")
        return

    roi = calib["gauge_roi"]
    model_calib = calib["model"]
    slope = model_calib["slope"]
    intercept = model_calib["intercept"]
    
    print(f"Menggunakan Tiang Virtual dari X={roi['x1']} sampai X={roi['x2']}")

    # 2. LOAD MODEL AI
    if not MODEL_PATH.exists():
        print(f"❌ Model tidak ditemukan di {MODEL_PATH}")
        return

    print("Memuat AI Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        encoder = "resnet18"

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

    # 3. BACA NAMA MEDIA
    config_video_path = CONFIG_PATH.parent / "current_media.txt"
    if config_video_path.exists():
        with open(config_video_path, "r") as f:
            media_name = f.read().strip()
    else:
        media_name = "12747554_1920_1080_24fps.mp4" # fallback
        
    media_path = TEST_VIDEOS_DIR / media_name
    is_video = media_path.suffix.lower() == ".mp4"
    
    if is_video:
        output_path = OUTPUT_DIR / f"measured_{media_name}"
    else:
        output_path = OUTPUT_DIR / f"measured_{media_path.stem}.jpg"

    if is_video:
        cap = cv2.VideoCapture(str(media_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Batasi frame agar tidak terlalu berat di CPU, tapi cukup panjang untuk melihat air naik (1500 frame ~ 1 menit)
        MAX_FRAMES = 1500
        if total_frames > MAX_FRAMES:
            total_frames = MAX_FRAMES
    else:
        frame_static = cv2.imread(str(media_path))
        if frame_static is None:
            print(f"❌ Gagal membaca gambar {media_path}")
            return
        height, width = frame_static.shape[:2]
        total_frames = 1
        out = None

    print(f"\nMemproses Video... (Maksimal {total_frames} frames)")
    
    # Ekstrapolasi Tiang Virtual dari ujung atas layar ke ujung bawah layar
    # Sehingga meskipun air melewati batas klik kalibrasi, AI tetap bisa mengukur!
    y_points = np.arange(0, height)
    
    # Hitung X untuk setiap Y (Persamaan Garis: X = x1 + (Y - y1) * dx/dy)
    if roi["y1"] == roi["y2"]:
        x_points = np.full_like(y_points, roi["x1"])
    else:
        dx = roi["x2"] - roi["x1"]
        dy = roi["y2"] - roi["y1"]
        x_points = roi["x1"] + (y_points - roi["y1"]) * dx / dy
        
    # Pastikan X tidak keluar dari batas gambar
    x_points = np.clip(np.round(x_points).astype(int), 0, width - 1)
    
    # Hitung ketinggian fisik (cm) untuk setiap titik pixel Y
    cm_points = slope * y_points + intercept

    use_amp = (device.type == "cuda")
    pbar = tqdm(total=total_frames, desc="Measuring")
    frame_count = 0
    
    kernel = np.ones((9, 9), np.uint8)

    while frame_count < total_frames:
        if is_video:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = frame_static.copy()

        frame_count += 1
        
        # --- AI SEGMENTATION ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        augmented = transform(image=frame_rgb)
        input_tensor = augmented["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(input_tensor)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                pred_mask = (prob > 0.5).astype(np.uint8)

        # Kembalikan mask ke resolusi asli
        pred_mask_resized = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        pred_mask_cleaned = cv2.morphologyEx(pred_mask_resized, cv2.MORPH_CLOSE, kernel)

        # --- PENGUKURAN TINGGI AIR ---
        current_water_level_cm = 0.0
        contact_point = None

        # Cari titik tertinggi dimana air menyentuh tiang virtual
        # Karena kita ingin air tertinggi, kita scan dari titik yang CM-nya paling besar ke kecil
        # Atau cukup cari indeks dimana mask == 1 di sepanjang garis tiang
        
        # Ekstrak nilai piksel mask di sepanjang garis tiang
        mask_values_on_line = pred_mask_cleaned[y_points, x_points]
        
        # Temukan semua titik sentuh (air)
        water_indices = np.where(mask_values_on_line == 1)[0]
        
        if len(water_indices) > 0:
            # Cari titik dengan nilai CM tertinggi
            valid_cms = cm_points[water_indices]
            max_idx_relative = np.argmax(valid_cms)
            max_idx_absolute = water_indices[max_idx_relative]
            
            current_water_level_cm = cm_points[max_idx_absolute]
            contact_point = (x_points[max_idx_absolute], y_points[max_idx_absolute])

        status_text, status_color = get_status(current_water_level_cm)

        # --- VISUALISASI EWS HUD ---
        overlay = frame.copy()
        
        # 1. Overlay Biru untuk Air
        water_area = pred_mask_cleaned == 1
        overlay[water_area, 0] = np.clip(overlay[water_area, 0] + 100, 0, 255)
        overlay[water_area, 1] = overlay[water_area, 1] * 0.7
        overlay[water_area, 2] = overlay[water_area, 2] * 0.7
        
        # 2. Gambar Tiang Virtual (Extrapolated Line)
        # Gambarkan dengan transparansi ringan atau putus-putus, tapi untuk mudahnya kita gambar garis penuh dari ujung ke ujung
        cv2.line(overlay, (x_points[0], y_points[0]), (x_points[-1], y_points[-1]), (200, 200, 200), 2)
        
        # Gambar garis tebal HANYA di area yang dikalibrasi manual oleh user
        cv2.line(overlay, (roi["x1"], roi["y1"]), (roi["x2"], roi["y2"]), (0, 255, 255), 4)
        
        # 3. Tandai Titik Sentuh Air (Lingkaran Merah)
        if contact_point:
            cv2.circle(overlay, contact_point, 8, status_color, -1)
            cv2.circle(overlay, contact_point, 10, (255, 255, 255), 2)
            
            # Tarik garis horizontal batas air
            cv2.line(overlay, (0, contact_point[1]), (width, contact_point[1]), status_color, 2)

        # 4. Kotak HUD Informasi
        cv2.rectangle(overlay, (20, 20), (500, 180), (0, 0, 0), -1)
        cv2.putText(overlay, f"RIVER EWS | VIRTUAL GAUGE", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, f"Level Air: {current_water_level_cm:.1f} cm", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(overlay, f"STATUS : {status_text}", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)

        if out:
            out.write(overlay)
        else:
            cv2.imwrite(str(output_path), overlay)
            
        pbar.update(1)

    pbar.close()
    if is_video:
        cap.release()
        out.release()
    
    print(f"\nSelesai! Hasil pengukuran disimpan di: {output_path}")

if __name__ == "__main__":
    main()
