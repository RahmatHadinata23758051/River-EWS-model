"""
Step 1.3 — Training DeepLabV3+ for Water Segmentation
=======================================================
Model  : DeepLabV3+ with ResNet-50 backbone (pretrained on ImageNet)
Dataset: RIWA v2 + Roboflow Water Detection (merged)
Loss   : 0.5 × BCEWithLogitsLoss + 0.5 × DiceLoss
Target : IoU > 95% on river test set

Usage:
    python 02_train_deeplabv3.py
    python 02_train_deeplabv3.py --epochs 100 --batch_size 4
    python 02_train_deeplabv3.py --encoder resnet101
"""

import argparse
import json
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm


def pil_load_rgb(path: Path) -> np.ndarray:
    """Load image as RGB numpy array via PIL — handles Windows long paths."""
    try:
        # PIL handles long paths on Windows better than cv2
        img = Image.open(str(path)).convert("RGB")
        return np.array(img)
    except Exception:
        return None


def pil_load_gray(path: Path) -> np.ndarray:
    """Load mask as grayscale numpy array via PIL — handles Windows long paths."""
    try:
        mask = Image.open(str(path)).convert("L")
        return np.array(mask)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent   # phase1_segmentation/
RIWA_DIR     = BASE_DIR.parent / "Segmentation" / "riwa_v2"
ROBOFLOW_DIR = BASE_DIR.parent / "Water detection.v21i.png-mask-semantic"
MODELS_DIR   = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG"}

DEFAULT_CFG = {
    "img_size":        512,
    "batch_size":      8,
    "epochs":          50,
    "lr":              1e-4,
    "weight_decay":    1e-4,
    "patience":        10,
    "num_workers":     0,        # 0 = safe on Windows
    "seed":            42,
    "encoder":         "resnet50",
    "encoder_weights": "imagenet",
    "threshold":       0.5,
}


# ──────────────────────────────────────────────────────────────
# DATASET — FORMAT A: RIWA (separate dirs, mask=255)
# ──────────────────────────────────────────────────────────────
class RIWADataset(Dataset):
    """RIWA-style: images/ and masks/ in separate folders. Mask value 255 = water."""

    def __init__(self, img_dir: Path, mask_dir: Path,
                 transform=None, split: str = "?"):
        self.transform = transform
        self.img_size  = 256  # fallback size for blank images

        imgs  = {f.stem.lower(): f for f in img_dir.iterdir()
                 if f.suffix in IMG_EXTS} if img_dir.exists() else {}
        masks = {f.stem.lower(): f for f in mask_dir.iterdir()
                 if f.suffix.lower() == ".png"} if mask_dir.exists() else {}

        self.pairs = [(imgs[s], masks[s]) for s in imgs if s in masks]
        self.pairs.sort(key=lambda x: x[0].name)
        print(f"  [RIWA/{split}] {len(self.pairs)} pairs loaded")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Use PIL for loading — handles Windows long paths (>260 chars)
        image = pil_load_rgb(img_path)
        mask_raw = pil_load_gray(mask_path)

        if image is None or mask_raw is None:
            # Fallback: blank tensors (skip corrupted files)
            blank = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            blank_m = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            if self.transform:
                aug = self.transform(image=blank, mask=blank_m)
                return aug["image"], aug["mask"].unsqueeze(0)
            return blank, blank_m

        mask = (mask_raw > 127).astype(np.float32)   # 255 → 1.0

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"].unsqueeze(0)
        return image, mask


# ──────────────────────────────────────────────────────────────
# DATASET — FORMAT B: Roboflow (flat folder, _mask suffix, mask=1)
# ──────────────────────────────────────────────────────────────
class RoboflowDataset(Dataset):
    """Roboflow flat-folder: image + image_mask.png in same dir. Mask value 1 = water."""

    def __init__(self, folder: Path, transform=None, split: str = "?"):
        self.transform = transform
        self.img_size  = 256  # fallback size for blank images

        if not folder.exists():
            self.pairs = []
            print(f"  [Roboflow/{split}] folder not found, skipping")
            return

        masks_raw = {f for f in folder.iterdir() if f.name.endswith("_mask.png")}
        imgs_raw  = {f for f in folder.iterdir()
                     if f.suffix.lower() in IMG_EXTS
                     and not f.name.endswith("_mask.png")}

        mask_dict = {f.stem[:-5].lower(): f for f in masks_raw}   # strip "_mask"
        img_dict  = {f.stem.lower(): f for f in imgs_raw}

        self.pairs = [(img_dict[s], mask_dict[s])
                      for s in img_dict if s in mask_dict]
        self.pairs.sort(key=lambda x: x[0].name)
        print(f"  [Roboflow/{split}] {len(self.pairs)} pairs loaded "
              f"({len(img_dict) - len(self.pairs)} images skipped, no mask)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Use PIL for loading — handles Windows long paths (>260 chars)
        image = pil_load_rgb(img_path)
        mask_raw = pil_load_gray(mask_path)

        if image is None or mask_raw is None:
            # Fallback: blank tensors
            blank   = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            blank_m = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            if self.transform:
                aug = self.transform(image=blank, mask=blank_m)
                return aug["image"], aug["mask"].unsqueeze(0)
            return blank, blank_m

        mask = (mask_raw > 0).astype(np.float32)   # 1 → 1.0

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"].unsqueeze(0)
        return image, mask


# ──────────────────────────────────────────────────────────────
# TRANSFORMS / AUGMENTATION
# ──────────────────────────────────────────────────────────────
def get_transforms(img_size: int, split: str):
    if split == "train":
        return A.Compose([
            # SmallestMaxSize ensures shortest side >= img_size → RandomCrop always fits
            A.SmallestMaxSize(max_size=img_size),
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.3),
            # Color augmentations — simulate different water/lighting conditions
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.05, p=0.7),
            A.GaussNoise(p=0.2),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            # Direct resize for val/test — deterministic, no padding issues
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


# ──────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ──────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = torch.sigmoid(pred).contiguous().view(-1)
        target = target.contiguous().view(-1)
        inter  = (pred * target).sum()
        return 1 - (2 * inter + self.smooth) / (
            pred.sum() + target.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce   = nn.BCEWithLogitsLoss()
        self.dice  = DiceLoss()
        self.bce_w = bce_weight

    def forward(self, pred, target):
        return self.bce_w * self.bce(pred, target) + \
               (1 - self.bce_w) * self.dice(pred, target)


# ──────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────
def compute_iou(logits, target, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum().item()
    union = (pred + target).clamp(0, 1).sum().item()
    return inter / (union + 1e-6)


def compute_dice(logits, target, thr=0.5):
    pred  = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum().item()
    return (2 * inter) / (pred.sum().item() + target.sum().item() + 1e-6)


# ──────────────────────────────────────────────────────────────
# TRAINING ENGINE
# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = total_iou = total_dice = 0.0

    for images, masks in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type,
                                enabled=(device.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_iou  += compute_iou(logits.detach(), masks)
        total_dice += compute_dice(logits.detach(), masks)

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = total_iou = total_dice = 0.0

    for images, masks in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, masks)

        total_loss += loss.item()
        total_iou  += compute_iou(logits, masks)
        total_dice += compute_dice(logits, masks)

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main(cfg: dict):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 65)
    print("  STEP 1.3 — TRAINING DeepLabV3+")
    print("  River EWS CV v2.0 | RIWA + Roboflow")
    print("=" * 65)
    print(f"  Device      : {device}")
    if device.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Encoder     : {cfg['encoder']} ({cfg['encoder_weights']})")
    print(f"  Image size  : {cfg['img_size']}×{cfg['img_size']}")
    print(f"  Batch size  : {cfg['batch_size']}")
    print(f"  Max epochs  : {cfg['epochs']}")
    print(f"  LR          : {cfg['lr']}")
    print(f"  Patience    : {cfg['patience']} epochs")

    tf_train = get_transforms(cfg["img_size"], "train")
    tf_val   = get_transforms(cfg["img_size"], "val")

    # ── TRAINING DATASET (RIWA train + Roboflow train merged) ──
    print("\n  Loading train datasets...")
    riwa_train = RIWADataset(
        RIWA_DIR / "images", RIWA_DIR / "masks",
        transform=tf_train, split="train"
    )
    rf_train = RoboflowDataset(
        ROBOFLOW_DIR / "train",
        transform=tf_train, split="train"
    )
    train_ds = ConcatDataset([riwa_train, rf_train])
    print(f"  ✅ Train total : {len(train_ds)} samples "
          f"({len(riwa_train)} RIWA + {len(rf_train)} Roboflow)")

    # ── VALIDATION DATASET (RIWA val only — clean reference) ──
    print("\n  Loading val dataset...")
    val_ds = RIWADataset(
        RIWA_DIR / "validation" / "images",
        RIWA_DIR / "validation" / "masks",
        transform=tf_val, split="val"
    )
    print(f"  ✅ Val total   : {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda")
    )

    # ── MODEL ─────────────────────────────────────────────────
    print(f"\n  Building DeepLabV3+ [{cfg['encoder']}]...")
    model = smp.DeepLabV3Plus(
        encoder_name=cfg["encoder"],
        encoder_weights=cfg["encoder_weights"],
        in_channels=3,
        classes=1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    criterion = CombinedLoss(bce_weight=0.5)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── TRAINING LOOP ─────────────────────────────────────────
    best_iou     = 0.0
    patience_cnt = 0
    history      = []

    print(f"\n  {'Epoch':>6} | {'Train Loss':>10} | {'Train IoU':>9} | "
          f"{'Val Loss':>8} | {'Val IoU':>7} | {'Val Dice':>8} | {'LR':>8}")
    print("  " + "─" * 72)

    t_start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_iou, tr_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        vl_loss, vl_iou, vl_dice = validate(
            model, val_loader, criterion, device)
        scheduler.step()

        lr_now  = scheduler.get_last_lr()[0]
        is_best = vl_iou > best_iou
        marker  = " ⭐" if is_best else ""

        print(f"  {epoch:6d} | {tr_loss:10.4f} | {tr_iou*100:8.2f}% | "
              f"{vl_loss:8.4f} | {vl_iou*100:6.2f}% | "
              f"{vl_dice*100:7.2f}% | {lr_now:.2e}{marker}")

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4), "train_iou": round(tr_iou, 4),
            "val_loss":   round(vl_loss, 4), "val_iou":   round(vl_iou, 4),
            "val_dice":   round(vl_dice, 4), "lr":        lr_now,
        })

        if is_best:
            best_iou     = vl_iou
            patience_cnt = 0
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou":              vl_iou,
                "val_dice":             vl_dice,
                "cfg":                  cfg,
            }, MODELS_DIR / "best_seg_model.pth")
            print(f"         → 💾 Model saved (IoU={vl_iou*100:.2f}%)")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                print(f"\n  ⏹️  Early stopping at epoch {epoch} "
                      f"(no improvement for {cfg['patience']} epochs)")
                break

    # Save last & log
    torch.save({
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "val_iou": vl_iou, "cfg": cfg,
    }, MODELS_DIR / "last_seg_model.pth")

    elapsed = time.time() - t_start
    log = {
        "timestamp":     datetime.now().isoformat(),
        "config":        cfg,
        "device":        str(device),
        "best_val_iou":  round(best_iou, 4),
        "total_epochs":  epoch,
        "elapsed_sec":   round(elapsed, 1),
        "train_samples": len(train_ds),
        "val_samples":   len(val_ds),
        "history":       history,
    }
    with open(MODELS_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # ── FINAL SUMMARY ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ✅ STEP 1.3 — TRAINING SELESAI")
    print("=" * 65)
    print(f"  Best Val IoU  : {best_iou * 100:.2f}%")
    print(f"  Total epochs  : {epoch}")
    print(f"  Training time : {elapsed / 60:.1f} menit")
    print(f"  Model saved   : models/best_seg_model.pth")

    target = 0.95
    if best_iou >= target:
        print(f"\n  🏆 TARGET TERCAPAI! IoU {best_iou*100:.2f}% ≥ {target*100:.0f}%")
        print("  ➡️  Lanjut ke: python 03_evaluate.py")
    else:
        print(f"\n  ⚠️  IoU {best_iou*100:.2f}% masih di bawah target {target*100:.0f}%")
        print("  Coba: --encoder resnet101, atau tambah epochs")
    print("=" * 65 + "\n")


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ — River EWS CV v2")
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_CFG["epochs"])
    parser.add_argument("--batch_size", type=int,   default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--lr",         type=float, default=DEFAULT_CFG["lr"])
    parser.add_argument("--img_size",   type=int,   default=DEFAULT_CFG["img_size"])
    parser.add_argument("--encoder",    type=str,   default=DEFAULT_CFG["encoder"],
                        choices=["resnet50", "resnet101",
                                 "efficientnet-b4", "mit_b2", "mit_b4"])
    parser.add_argument("--patience",   type=int,   default=DEFAULT_CFG["patience"])
    parser.add_argument("--batch_size_override", type=int, default=None,
                        help="Override batch size (gunakan ini jika OOM)")
    args = parser.parse_args()

    cfg = {**DEFAULT_CFG, **vars(args)}
    if cfg.get("batch_size_override"):
        cfg["batch_size"] = cfg["batch_size_override"]

    main(cfg)
