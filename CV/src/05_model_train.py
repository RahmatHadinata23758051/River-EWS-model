"""
Training Pipeline
Training script untuk U-Net water segmentation model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import os

import sys
sys.path.insert(0, str(Path(__file__).parent))

# Import dari 04_model_unet_architecture.py
import importlib.util
spec = importlib.util.spec_from_file_location("unet_model", Path(__file__).parent / "04_model_unet_architecture.py")
unet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet_module)
create_model = unet_module.create_model
count_parameters = unet_module.count_parameters


class FloodSegmentationDataset(Dataset):
    """Dataset loader untuk Flood Segmentation"""
    
    def __init__(self, image_dir, mask_dir, image_size=256, transforms=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.transforms = transforms
        
        # Collect all image-mask pairs
        self.samples = []
        
        for subfolder in sorted(self.image_dir.iterdir()):
            if not subfolder.is_dir():
                continue
            
            # Find corresponding mask subfolder
            mask_subfolder = self.mask_dir / subfolder.name
            if not mask_subfolder.exists():
                continue
            
            # Collect image-mask pairs
            for img_file in subfolder.glob("*.jpg"):
                mask_file = mask_subfolder / f"{img_file.stem}_binary.png"
                if mask_file.exists():
                    self.samples.append((img_file, mask_file))
        
        print(f"Loaded {len(self.samples)} image-mask pairs from {image_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Resize
        image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # Normalize
        image_rgb = image_rgb.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'path': str(img_path)
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Calculate IoU
            pred_binary = (outputs > 0.5).float()
            intersection = torch.sum(pred_binary * masks)
            union = torch.sum(pred_binary) + torch.sum(masks) - intersection
            iou = intersection / (union + 1e-6)
            total_iou += iou.item()
    
    return total_loss / len(dataloader), total_iou / len(dataloader)


def main():
    """Main training pipeline"""
    
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 8,
        'learning_rate': 0.001,
        'epochs': 20,
        'image_size': 256,
        'checkpoint_dir': '../model',
    }
    
    print("="*80)
    print("U-NET TRAINING PIPELINE - WATER SEGMENTATION")
    print("="*80)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['epochs']}")
    
    # Create checkpoint directory
    Path(config['checkpoint_dir']).mkdir(exist_ok=True)
    
    # Create dataset
    print("\n[1] LOADING DATASET")
    try:
        dataset = FloodSegmentationDataset(
            image_dir='../data/images',
            mask_dir='../data/binary_masks',
            image_size=config['image_size']
        )
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("Make sure binary_masks folder exists. Run 03_create_binary_masks.py first.")
        return
    
    if len(dataset) < 10:
        print("⚠️ Warning: Dataset has fewer than 10 samples. Consider preparing more data.")
        train_size = max(1, len(dataset) // 2)
        val_size = len(dataset) - train_size
    else:
        train_size = int(len(dataset) * 0.7)
        val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print("\n[2] CREATING MODEL")
    model = create_model(device=config['device'])
    params = count_parameters(model)
    print(f"Trainable parameters: {params:,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    print("\n[3] TRAINING")
    print("-"*80)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': []
    }
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config['device'])
        history['train_loss'].append(train_loss)
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, config['device'])
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val IoU: {val_iou:.6f}")
        
        # Save checkpoint
        checkpoint_path = Path(config['checkpoint_dir']) / f"model_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
        }, checkpoint_path)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_path = Path(config['checkpoint_dir']) / 'best_model.pth'
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best model saved to {best_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print("\n[!] Early stopping triggered")
                break
        
        # Learning rate scheduler
        scheduler.step(val_loss)
    
    # Save final results
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    # Save history
    history_path = Path(config['checkpoint_dir']) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Summary
    print(f"\nBest Val Loss: {min(history['val_loss']):.6f}")
    print(f"Best Val IoU: {max(history['val_iou']):.6f}")
    print(f"\nCheckpoints saved in: {Path(config['checkpoint_dir']).absolute()}")


if __name__ == "__main__":
    main()
