"""
External Video Segmentation Script
Output: Video segmentasi dengan overlay water detection
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import time
import json
from datetime import datetime
import importlib.util

# Import U-Net
spec = importlib.util.spec_from_file_location(
    "unet_model",
    Path(__file__).parent / "04_model_unet_architecture.py"
)
unet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet_module)
create_model = unet_module.create_model


def load_model(model_path, device):
    model = create_model(device=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def get_status(water_pct):
    if water_pct < 5:
        return "AMAN", (76, 175, 80)        # Green
    elif water_pct < 15:
        return "SIAGA", (0, 200, 255)       # Yellow (BGR)
    elif water_pct < 30:
        return "WASPADA", (0, 140, 255)     # Orange (BGR)
    else:
        return "BAHAYA", (50, 50, 220)      # Red (BGR)


def detect_frame(model, frame, device, threshold=0.5):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (256, 256))
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        output_np = output.cpu().numpy()[0, 0]

    mask = cv2.resize(output_np, (w, h), interpolation=cv2.INTER_LINEAR)
    binary = (mask > threshold).astype(np.uint8) * 255
    water_pct = np.sum(mask > threshold) / mask.size * 100
    return mask, binary, water_pct


def draw_overlay(frame, binary_mask, water_pct):
    """Draw water overlay + HUD on frame"""
    status_text, color_bgr = get_status(water_pct)

    # --- Water color overlay ---
    overlay = frame.copy()
    water_region = binary_mask == 255
    overlay[water_region] = (
        overlay[water_region] * 0.4 +
        np.array(color_bgr, dtype=np.float32) * 0.6
    ).astype(np.uint8)

    # Blend original + overlay
    result = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

    h, w = result.shape[:2]

    # --- HUD panel (top-left) ---
    panel_h = 110
    panel_w = 380
    panel = result[0:panel_h, 0:panel_w].copy()
    # Dark transparent background
    dark = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    result[0:panel_h, 0:panel_w] = cv2.addWeighted(panel, 0.3, dark, 0.7, 0)

    # Status indicator circle
    cv2.circle(result, (35, 35), 20, color_bgr, -1)
    cv2.circle(result, (35, 35), 20, (255, 255, 255), 2)

    # Status text
    cv2.putText(result, status_text,
                (65, 28), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # Water percentage bar
    bar_x, bar_y, bar_w, bar_h = 15, 55, 350, 18
    cv2.rectangle(result, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill_w = int(bar_w * min(water_pct / 100, 1.0))
    if fill_w > 0:
        cv2.rectangle(result, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color_bgr, -1)
    cv2.rectangle(result, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

    # Water % text
    cv2.putText(result, f"Water: {water_pct:.1f}%",
                (bar_x, bar_y + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Threshold markers on bar
    for threshold_val, label in [(5, ""), (15, ""), (30, "")]:
        marker_x = bar_x + int(bar_w * threshold_val / 100)
        cv2.line(result, (marker_x, bar_y - 3), (marker_x, bar_y + bar_h + 3), (255, 255, 255), 1)

    # --- EWS label (bottom-right) ---
    cv2.putText(result, "River-EWS | U-Net Flood Detection",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return result


def process_video_to_segmentation(model, video_path, output_path, device):
    """Process video → output segmentation video"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ❌ Cannot open: {video_path.name}")
        return None

    fps     = cap.get(cv2.CAP_PROP_FPS)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur     = total / fps if fps > 0 else 0

    # Resize output so it's not too huge (max 1280px wide)
    scale = min(1280 / w_orig, 1.0)
    out_w  = int(w_orig * scale) & ~1   # must be even
    out_h  = int(h_orig * scale) & ~1

    print(f"  📹 Input:  {w_orig}x{h_orig} @ {fps:.1f}FPS  ({dur:.1f}s, {total} frames)")
    print(f"  📤 Output: {out_w}x{out_h}  →  {output_path.name}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    if not out.isOpened():
        print("  ❌ Cannot create output video writer")
        cap.release()
        return None

    water_pcts  = []
    statuses    = []
    frame_idx   = 0
    t_start     = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Resize frame to output size
        frame_resized = cv2.resize(frame, (out_w, out_h))

        # Detect water
        _, binary, water_pct = detect_frame(model, frame_resized, device)
        status, _ = get_status(water_pct)

        water_pcts.append(water_pct)
        statuses.append(status)

        # Draw overlay and write
        frame_out = draw_overlay(frame_resized, binary, water_pct)
        out.write(frame_out)

        # Progress
        if frame_idx % 60 == 0 or frame_idx == total:
            elapsed = time.time() - t_start
            pct_done = frame_idx / total * 100
            eta = (elapsed / frame_idx) * (total - frame_idx) if frame_idx > 0 else 0
            print(f"    [{pct_done:5.1f}%] Frame {frame_idx}/{total} | "
                  f"{status} ({water_pct:.1f}%) | ETA: {eta:.0f}s")

    cap.release()
    out.release()

    elapsed = time.time() - t_start
    fps_actual = frame_idx / elapsed if elapsed > 0 else 0

    result = {
        'video_name':    video_path.name,
        'output_name':   output_path.name,
        'resolution_in': f"{w_orig}x{h_orig}",
        'resolution_out': f"{out_w}x{out_h}",
        'fps':           fps,
        'total_frames':  frame_idx,
        'duration_sec':  dur,
        'avg_water_pct': float(np.mean(water_pcts)) if water_pcts else 0,
        'max_water_pct': float(np.max(water_pcts)) if water_pcts else 0,
        'min_water_pct': float(np.min(water_pcts)) if water_pcts else 0,
        'process_time_sec': round(elapsed, 2),
        'process_fps':   round(fps_actual, 2),
        'status_counts': {
            s: statuses.count(s)
            for s in ['Aman', 'Siaga', 'Waspada', 'Bahaya']
        },
        'dominant_status': max(set(statuses), key=statuses.count) if statuses else 'N/A',
    }

    size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
    print(f"\n  ✅ Selesai! Output: {output_path.name} ({size_mb:.1f} MB)")
    print(f"  ⏱️  Processing: {elapsed:.1f}s ({fps_actual:.2f} fps)")
    print(f"  💧 Water avg: {result['avg_water_pct']:.2f}%  max: {result['max_water_pct']:.2f}%")
    print(f"  🏆 Dominant status: {result['dominant_status']}")

    return result


def main():
    print("=" * 80)
    print("  RIVER EWS — U-Net VIDEO SEGMENTATION")
    print("=" * 80)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}\n")

    # Load model
    model_path = Path(__file__).parent.parent / "model" / "best_model.pth"
    print(f"  Loading model: {model_path.name} ...")
    model = load_model(str(model_path), device)
    print(f"  ✅ Model ready!\n")

    # Find test videos
    test_dir = Path(__file__).parent.parent / "test-video"
    if not test_dir.exists():
        print(f"  ❌ Folder tidak ditemukan: {test_dir}")
        return

    videos = sorted(
        list(test_dir.glob('*.mp4')) +
        list(test_dir.glob('*.avi')) +
        list(test_dir.glob('*.mov'))
    )

    if not videos:
        print(f"  ❌ Tidak ada video di: {test_dir}")
        return

    print(f"  Ditemukan {len(videos)} video:")
    for v in videos:
        print(f"    📁 {v.name}  ({v.stat().st_size/1024/1024:.1f} MB)")

    # Output folder
    out_dir = test_dir / "segmented"
    out_dir.mkdir(exist_ok=True)
    print(f"\n  Output folder: {out_dir}\n")

    # Process each video
    all_results = []
    for i, vp in enumerate(videos):
        print(f"{'─'*80}")
        print(f"  [{i+1}/{len(videos)}] {vp.name}")
        print(f"{'─'*80}")

        out_name = vp.stem + "_segmented.mp4"
        out_path = out_dir / out_name

        result = process_video_to_segmentation(model, vp, out_path, device)
        if result:
            all_results.append(result)

    # Save JSON summary
    report = {
        'date': datetime.now().isoformat(),
        'device': device,
        'model': 'best_model.pth (U-Net, IoU=94.08%)',
        'videos': all_results
    }
    report_path = out_dir / "segmentation_report.json"
    with open(str(report_path), 'w') as f:
        json.dump(report, f, indent=2)

    # Final summary
    print(f"\n{'='*80}")
    print(f"  ✅ SELESAI — {len(all_results)} video tersegmentasi")
    print(f"{'='*80}")
    for r in all_results:
        icon = {"Aman":"🟢","Siaga":"🟡","Waspada":"🟠","Bahaya":"🔴"}.get(r['dominant_status'], "⚪")
        print(f"  {icon}  {r['output_name']}")
        print(f"      Water avg: {r['avg_water_pct']:.1f}%  max: {r['max_water_pct']:.1f}%  "
              f"status: {r['dominant_status']}  speed: {r['process_fps']:.2f}fps")
    print(f"\n  📁 Semua output di: {out_dir.absolute()}")
    print(f"  📄 Report: {report_path.name}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
