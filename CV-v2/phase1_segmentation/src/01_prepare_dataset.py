"""
Step 1.2 — Prepare & Verify Dataset
====================================
Mendukung DUA format dataset:

FORMAT A — RIWA v2 (images/ dan masks/ terpisah)
  • masks bernilai 0 (background) dan 255 (water)
  • struktur: images/*.jpg, masks/*.png

FORMAT B — Roboflow (images + masks dalam 1 folder)
  • masks bernilai 0 (background) dan 1 (water)
  • nama mask = {nama_image}_mask.png
  • struktur: train/*.jpg + train/*_mask.png

Script ini:
1. Otomatis deteksi format dataset
2. Verifikasi jumlah image = jumlah mask
3. Tampilkan statistik & resolusi
4. Visualisasi sample dengan overlay biru di area air
5. Konfirmasi dataset siap untuk training
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────
# PATH CONFIGURATION
# ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
OUTPUT_DIR  = BASE_DIR / "data_check"
OUTPUT_DIR.mkdir(exist_ok=True)

# Dataset roots
RIWA_DIR     = BASE_DIR.parent / "Segmentation" / "riwa_v2"
ROBOFLOW_DIR = BASE_DIR.parent / "Water detection.v21i.png-mask-semantic"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG"}


# ──────────────────────────────────────────────
# FORMAT A — RIWA (separate images/ masks/ dirs)
# ──────────────────────────────────────────────
def scan_riwa_split(split_name: str, img_dir: Path, mask_dir: Path) -> dict:
    """Scan RIWA-style split. mask value 255 = water."""
    images = {f.stem.lower(): f for f in img_dir.iterdir()
              if f.suffix.lower() in IMG_EXTS} if img_dir.exists() else {}
    masks  = {f.stem.lower(): f for f in mask_dir.iterdir()
              if f.suffix.lower() == ".png"}  if mask_dir.exists() else {}

    matched   = [s for s in images if s in masks]
    img_only  = [s for s in images if s not in masks]
    mask_only = [s for s in masks  if s not in images]

    widths, heights = sample_resolutions(matched, images)
    return _build_stats(split_name, images, masks, matched,
                        img_only, mask_only, widths, heights,
                        mask_format="255")


# ──────────────────────────────────────────────
# FORMAT B — Roboflow (flat folder, _mask suffix)
# ──────────────────────────────────────────────
def scan_roboflow_split(split_name: str, folder: Path) -> dict:
    """Scan Roboflow flat folder. mask value 1 = water."""
    if not folder.exists():
        return _build_stats(split_name, {}, {}, [], [], [], [], [], "1")

    all_files = list(folder.iterdir())

    # Masks = files ending in _mask.png
    masks_raw = {f for f in all_files if f.name.endswith("_mask.png")}
    # Images  = everything else that's an image
    imgs_raw  = {f for f in all_files
                 if f.suffix.lower() in IMG_EXTS
                 and not f.name.endswith("_mask.png")}

    # Build lookup: stem_without_mask → Path
    mask_dict = {f.stem[:-5].lower(): f for f in masks_raw}   # remove "_mask"
    img_dict  = {f.stem.lower(): f for f in imgs_raw}

    matched   = [s for s in img_dict if s in mask_dict]
    img_only  = [s for s in img_dict if s not in mask_dict]
    mask_only = [s for s in mask_dict if s not in img_dict]

    widths, heights = sample_resolutions(matched, img_dict)
    return _build_stats(split_name, img_dict, mask_dict, matched,
                        img_only, mask_only, widths, heights,
                        mask_format="1")


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def sample_resolutions(matched: list, img_dict: dict, n: int = 50):
    widths, heights = [], []
    for s in matched[:n]:
        try:
            with Image.open(img_dict[s]) as im:
                widths.append(im.width)
                heights.append(im.height)
        except Exception:
            pass
    return widths, heights


def _build_stats(split_name, images, masks, matched,
                 img_only, mask_only, widths, heights, mask_format):
    return {
        "split":        split_name,
        "total_images": len(images),
        "total_masks":  len(masks),
        "matched_pairs": len(matched),
        "img_no_mask":  len(img_only),
        "mask_no_img":  len(mask_only),
        "matched_list": matched,
        "img_stems":    images,
        "mask_stems":   masks,
        "mask_format":  mask_format,   # "1" or "255"
        "avg_width":    int(np.mean(widths))  if widths else 0,
        "avg_height":   int(np.mean(heights)) if heights else 0,
        "min_width":    int(np.min(widths))   if widths else 0,
        "min_height":   int(np.min(heights))  if heights else 0,
    }


def load_mask_binary(mask_path: Path, mask_format: str) -> np.ndarray:
    """Load mask and convert to binary 0/1 array regardless of source format."""
    mask = np.array(Image.open(mask_path).convert("L"))
    if mask_format == "1":
        return (mask > 0).astype(np.uint8)    # Roboflow: 1 = water
    else:
        return (mask > 127).astype(np.uint8)  # RIWA: 255 = water


def visualize_samples(dataset_name: str, img_stems: dict, mask_stems: dict,
                      matched: list, mask_format: str, n: int = 6):
    """Save visualization grid: original | mask | overlay."""
    if not matched:
        return

    samples = random.sample(matched, min(n, len(matched)))
    fig, axes = plt.subplots(len(samples), 3,
                             figsize=(12, len(samples) * 3))
    fig.suptitle(f"{dataset_name} — Sample Verification",
                 fontsize=13, fontweight='bold')

    for r, s in enumerate(samples):
        try:
            img      = np.array(Image.open(img_stems[s]).convert("RGB"))
            mask_bin = load_mask_binary(mask_stems[s], mask_format)
        except Exception as e:
            print(f"  ⚠️ Cannot open {s}: {e}")
            continue

        overlay = img.copy()
        water_px = mask_bin == 1
        overlay[water_px] = (
            overlay[water_px] * 0.4 + np.array([20, 80, 220]) * 0.6
        ).clip(0, 255).astype(np.uint8)
        water_pct = water_px.mean() * 100

        row = axes[r] if len(samples) > 1 else axes
        row[0].imshow(img);        row[0].axis('off')
        row[1].imshow(mask_bin, cmap='Blues', vmin=0, vmax=1); row[1].axis('off')
        row[2].imshow(overlay);    row[2].axis('off')

        if r == 0:
            row[0].set_title("Image",                 fontsize=9, fontweight='bold')
            row[1].set_title("Binary Mask (water=1)", fontsize=9, fontweight='bold')
            row[2].set_title("Overlay (blue=water)",  fontsize=9, fontweight='bold')
        else:
            row[2].set_title(f"Water: {water_pct:.1f}%", fontsize=8,
                             color='navy' if water_pct > 5 else 'gray')

    plt.tight_layout()
    safe_name = dataset_name.replace(" ", "_").replace("/", "_")
    out_path = OUTPUT_DIR / f"samples_{safe_name}.png"
    fig.savefig(str(out_path), dpi=90, bbox_inches='tight')
    plt.close(fig)
    print(f"  📸 Visualization → {out_path.name}")


def print_banner(text: str):
    line = "─" * 62
    print(f"\n{line}\n  {text}\n{line}")


def print_stats(stats: dict):
    ok = "✅" if stats["img_no_mask"] == 0 and stats["mask_no_img"] == 0 else "⚠️ "
    print(f"  {ok} Images         : {stats['total_images']}")
    print(f"  {ok} Masks          : {stats['total_masks']}")
    print(f"  ✅ Matched pairs   : {stats['matched_pairs']}")
    if stats["img_no_mask"]:
        print(f"  ⚠️  Images w/o mask: {stats['img_no_mask']}")
    if stats["mask_no_img"]:
        print(f"  ⚠️  Masks w/o image: {stats['mask_no_img']}")
    if stats["avg_width"]:
        print(f"  📐 Avg resolution  : {stats['avg_width']} × {stats['avg_height']} px")
    print(f"  🏷️  Mask format    : pixel value = {stats['mask_format']} (water)")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("\n" + "=" * 62)
    print("  STEP 1.2 — DATASET VERIFICATION")
    print("  River EWS CV v2.0 | Multi-Dataset")
    print("=" * 62)

    all_stats   = {}
    total_pairs = 0

    # ── DATASET 1: RIWA v2 ────────────────────────────────────
    print_banner("DATASET 1: RIWA v2 (FORMAT A — separate dirs)")
    print(f"  Path: {RIWA_DIR}")

    if RIWA_DIR.exists():
        for split_name, dirs in [
            ("riwa_train", (RIWA_DIR / "images",              RIWA_DIR / "masks")),
            ("riwa_val",   (RIWA_DIR / "validation" / "images", RIWA_DIR / "validation" / "masks")),
            ("riwa_test",  (RIWA_DIR / "test" / "images",      RIWA_DIR / "test" / "masks")),
        ]:
            img_dir, mask_dir = dirs
            if not img_dir.exists():
                print(f"  ⏭️  Skipping {split_name} (not found)")
                continue
            stats = scan_riwa_split(split_name, img_dir, mask_dir)
            all_stats[split_name] = stats
            total_pairs += stats["matched_pairs"]
            print(f"\n  [{split_name}]")
            print_stats(stats)
            if stats["matched_pairs"] > 0:
                visualize_samples(split_name,
                                  stats["img_stems"], stats["mask_stems"],
                                  stats["matched_list"], stats["mask_format"])
    else:
        print(f"  ❌ RIWA tidak ditemukan: {RIWA_DIR}")

    # ── DATASET 2: Roboflow ───────────────────────────────────
    print_banner("DATASET 2: Roboflow Water Detection (FORMAT B — flat folder)")
    print(f"  Path: {ROBOFLOW_DIR}")

    if ROBOFLOW_DIR.exists():
        for split_name, folder in [
            ("roboflow_train", ROBOFLOW_DIR / "train"),
            ("roboflow_val",   ROBOFLOW_DIR / "valid"),
            ("roboflow_test",  ROBOFLOW_DIR / "test"),
        ]:
            if not folder.exists():
                print(f"  ⏭️  Skipping {split_name} (folder not found: {folder.name})")
                continue
            stats = scan_roboflow_split(split_name, folder)
            all_stats[split_name] = stats
            total_pairs += stats["matched_pairs"]
            print(f"\n  [{split_name}] — {folder}")
            print_stats(stats)
            if stats["matched_pairs"] > 0:
                visualize_samples(split_name,
                                  stats["img_stems"], stats["mask_stems"],
                                  stats["matched_list"], stats["mask_format"])
    else:
        print(f"  ❌ Roboflow folder tidak ditemukan: {ROBOFLOW_DIR}")

    # ── Grand Summary ─────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  GRAND SUMMARY")
    print("=" * 62)
    for name, s in all_stats.items():
        fmt = f"[mask={s['mask_format']}]"
        print(f"  {name:20s} → {s['matched_pairs']:5d} pairs  {fmt}")

    print(f"\n  {'TOTAL':20s} → {total_pairs:5d} pairs")

    issues = [f"{k}: {v['img_no_mask']} images tanpa mask"
              for k, v in all_stats.items() if v["img_no_mask"] > 0]
    issues += [f"{k}: 0 matched pairs!" for k, v in all_stats.items()
               if v["matched_pairs"] == 0]

    print()
    if issues:
        print("  ⚠️  ISSUES:")
        for i in issues:
            print(f"     - {i}")
    else:
        print("  ✅ Semua dataset OK — tidak ada masalah!")
        print("  ✅ Siap untuk training DeepLabV3+")

    # Save JSON
    stats_out = {k: {kk: vv for kk, vv in v.items()
                     if kk not in ("matched_list", "img_stems", "mask_stems")}
                 for k, v in all_stats.items()}
    stats_out["total_pairs"] = total_pairs

    json_path = OUTPUT_DIR / "dataset_stats.json"
    with open(str(json_path), "w") as f:
        json.dump(stats_out, f, indent=2)

    print(f"\n  📄 Stats → {json_path}")
    print("\n" + "=" * 62)
    print("  ✅ STEP 1.2 SELESAI")
    print("  Cek visualisasi di: phase1_segmentation/data_check/")
    print("  Lanjut ke: python 02_train_deeplabv3.py")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
