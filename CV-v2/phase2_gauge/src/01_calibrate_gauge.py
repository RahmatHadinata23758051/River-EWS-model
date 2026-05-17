import os
import cv2
import json
import numpy as np
from pathlib import Path

# --- KONFIGURASI PATH ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEST_VIDEOS_DIR = BASE_DIR.parent / "CV" / "test-video"
CONFIG_DIR = BASE_DIR / "phase3_integration" / "configs"
CONFIG_PATH = CONFIG_DIR / "calibration.json"

# Pastikan folder config ada
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Variabel Global untuk GUI OpenCV dan image processing
points = []
frame_display = None
frame_clean = None

def mouse_callback(event, x, y, flags, param):
    global points, frame_display, frame_clean
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            # Gambar titik
            cv2.circle(frame_display, (x, y), 5, (0, 0, 255), -1)
            
            # Jika sudah ada 2 titik, gambar garis
            if len(points) == 2:
                cv2.line(frame_display, points[0], points[1], (0, 255, 255), 2)
            
            cv2.imshow("Kalibrasi Tiang Virtual", frame_display)

def main():
    global points, frame_display, frame_clean
    
    # Cari semua video dan gambar di folder test-video
    media_files = []
    for ext in ["*.mp4", "*.jpg", "*.jpeg", "*.png", "*.jfif"]:
        media_files.extend(list(TEST_VIDEOS_DIR.glob(ext)))
        
    if not media_files:
        print(f"❌ Tidak ada file media (.mp4/.jpg/.png/.jfif) di {TEST_VIDEOS_DIR}")
        return
        
    print(f"\nPilih media untuk dikalibrasi:")
    for i, file_path in enumerate(media_files):
        print(f"[{i}] {file_path.name}")
        
    try:
        idx = int(input("\nMasukkan nomor pilihan: "))
        selected_media = media_files[idx]
    except (ValueError, IndexError):
        print("❌ Pilihan tidak valid!")
        return

    # Simpan nama media yang dipilih
    with open(CONFIG_DIR / "current_media.txt", "w") as f:
        f.write(selected_media.name)

    # Ambil frame pertama
    if selected_media.suffix.lower() == ".mp4":
        cap = cv2.VideoCapture(str(selected_media))
        ret, frame = cap.read()
        cap.release()
    else:
        frame = cv2.imread(str(selected_media))
        ret = frame is not None
    
    if not ret:
        print("❌ Gagal membaca frame dari video!")
        return

    # Resize agar muat di layar laptop standar saat kalibrasi
    # (Kita akan mencatat rasio resize agar koordinat tetap valid untuk resolusi asli)
    orig_h, orig_w = frame.shape[:2]
    display_w = min(1280, orig_w)
    display_h = int((display_w / orig_w) * orig_h)
    scale_ratio = orig_w / display_w
    
    frame_display = cv2.resize(frame, (display_w, display_h))
    frame_clean = frame_display.copy()
    
    print("\n" + "="*50)
    print(" STEP 2.1 — KALIBRASI TIANG VIRTUAL (PEILSCHAAL)")
    print("="*50)
    print("INSTRUKSI:")
    print("1. Jendela video akan terbuka.")
    print("2. Klik KIRI pada titik DASAR sungai (Ketinggian 0 cm).")
    print("3. Klik KIRI lagi pada titik ATAS tebing (Ketinggian Maksimal).")
    print("4. Tekan tombol 'ENTER' di keyboard jika sudah puas, atau 'r' untuk reset titik.")
    print("="*50)

    cv2.namedWindow("Kalibrasi Tiang Virtual")
    cv2.setMouseCallback("Kalibrasi Tiang Virtual", mouse_callback)

    while True:
        cv2.imshow("Kalibrasi Tiang Virtual", frame_display)
        key = cv2.waitKey(1) & 0xFF
        
        # Tekan 'r' untuk reset
        if key == ord('r'):
            points = []
            frame_display = frame_clean.copy()
            print("Titik di-reset. Silakan klik ulang 2 titik.")
            
        # Tekan ENTER (13) untuk konfirmasi
        elif key == 13: 
            if len(points) == 2:
                break
            else:
                print("⚠️ Anda harus memilih tepat 2 titik sebelum menekan ENTER!")
        
        # Tekan ESC (27) atau 'q' untuk keluar
        elif key == 27 or key == ord('q'):
            print("Kalibrasi dibatalkan oleh pengguna.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    # Kembalikan koordinat ke resolusi asli
    pt1_orig = (int(points[0][0] * scale_ratio), int(points[0][1] * scale_ratio))
    pt2_orig = (int(points[1][0] * scale_ratio), int(points[1][1] * scale_ratio))

    print(f"\nTitik 1 (Original Res): X={pt1_orig[0]}, Y={pt1_orig[1]}")
    print(f"Titik 2 (Original Res): X={pt2_orig[0]}, Y={pt2_orig[1]}")

    # Meminta input tinggi dalam dunia nyata dari user via terminal
    try:
        val1 = float(input(f"Masukkan ketinggian (cm) untuk Titik 1 (Biasanya 0): "))
        val2 = float(input(f"Masukkan ketinggian (cm) untuk Titik 2 (Misal 300): "))
    except ValueError:
        print("❌ Input harus berupa angka! Kalibrasi dibatalkan.")
        return

    # Hitung Regresi Linear: Ketinggian(cm) = slope * Y_pixel + intercept
    # Perhatikan: Di komputer, Y=0 ada di atas, Y max ada di bawah.
    y1, y2 = pt1_orig[1], pt2_orig[1]
    
    if y1 == y2:
        print("❌ Titik 1 dan Titik 2 tidak boleh berada pada garis horizontal yang persis sama!")
        return

    slope = (val2 - val1) / (y2 - y1)
    intercept = val1 - (slope * y1)

    print("\n--- HASIL KALIBRASI ---")
    print(f"Setiap 1 pixel bergerak vertikal mewakili {abs(slope):.2f} cm.")

    # Susun data JSON
    calib_data = {
        "_comment": "Kalibrasi pixel ke cm untuk River EWS",
        "camera_id": "river_cam_01",
        "calibration_date": "2026-05-16",
        "image_resolution": {"width": orig_w, "height": orig_h},
        "gauge_roi": {
            "x1": pt1_orig[0], "y1": pt1_orig[1],
            "x2": pt2_orig[0], "y2": pt2_orig[1]
        },
        "calibration_points": [
            {"pixel_y": pt1_orig[1], "real_cm": val1},
            {"pixel_y": pt2_orig[1], "real_cm": val2}
        ],
        "model": {
            "type": "linear",
            "slope": slope,
            "intercept": intercept,
            "r_squared": 1.0 # Perfect fit for 2 points
        },
        "status": "CALIBRATED"
    }

    # Simpan ke file
    with open(CONFIG_PATH, "w") as f:
        json.dump(calib_data, f, indent=2)

    print(f"\n✅ Kalibrasi berhasil disimpan di: {CONFIG_PATH}")
    print("Anda sekarang bisa menjalankan skrip 02_measure_water_level.py!")

if __name__ == "__main__":
    main()
