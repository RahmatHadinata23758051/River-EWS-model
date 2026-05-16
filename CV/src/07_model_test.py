"""
Comprehensive Model Testing Script
Evaluates the trained U-Net model on test data with full metrics and visualizations
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
import importlib.util

# Import U-Net architecture
spec = importlib.util.spec_from_file_location(
    "unet_model", 
    Path(__file__).parent / "04_model_unet_architecture.py"
)
unet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet_module)
create_model = unet_module.create_model
count_parameters = unet_module.count_parameters


def load_model(model_path, device):
    """Load trained model"""
    model = create_model(device=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_test_samples(image_dir, mask_dir, max_samples=300):
    """Load image-mask pairs for testing"""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    samples = []
    
    for subfolder in sorted(image_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        mask_subfolder = mask_dir / subfolder.name
        if not mask_subfolder.exists():
            continue
        for img_file in sorted(subfolder.glob("*.jpg")):
            mask_file = mask_subfolder / f"{img_file.stem}_binary.png"
            if mask_file.exists():
                samples.append((img_file, mask_file))
            if len(samples) >= max_samples:
                break
        if len(samples) >= max_samples:
            break
    
    return samples


def predict_single(model, image_path, device, image_size=256):
    """Run inference on a single image"""
    image = cv2.imread(str(image_path))
    original_h, original_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    resized = cv2.resize(image_rgb, (image_size, image_size))
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        output_np = output.cpu().numpy()[0, 0]
    
    mask = cv2.resize(output_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    return image_rgb, mask


def calculate_metrics(pred_mask, gt_mask, threshold=0.5):
    """Calculate segmentation metrics"""
    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = (gt_mask > 0.5).astype(np.float32)
    
    tp = np.sum(pred_binary * gt_binary)
    fp = np.sum(pred_binary * (1 - gt_binary))
    fn = np.sum((1 - pred_binary) * gt_binary)
    tn = np.sum((1 - pred_binary) * (1 - gt_binary))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + 1e-8)
    
    dice = 2 * intersection / (2 * intersection + fp + fn + 1e-8)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }


def get_flood_status(water_pct):
    """Classify flood status"""
    if water_pct < 5:
        return "Aman", "🟢"
    elif water_pct < 15:
        return "Siaga", "🟡"
    elif water_pct < 30:
        return "Waspada", "🟠"
    else:
        return "Bahaya", "🔴"


def run_test():
    """Main test pipeline"""
    
    print("=" * 80)
    print("  COMPREHENSIVE U-NET MODEL TEST - FLOOD WATER SEGMENTATION")
    print("=" * 80)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # === 1. LOAD MODEL ===
    print("\n" + "─" * 80)
    print("  [1/6] LOADING MODEL")
    print("─" * 80)
    
    model_path = Path(__file__).parent / '../model/best_model.pth'
    if not model_path.exists():
        print(f"  ❌ Model not found: {model_path}")
        return
    
    model = load_model(str(model_path), device)
    params = count_parameters(model)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    
    print(f"  ✅ Model loaded: {model_path.name}")
    print(f"  📊 Parameters: {params:,}")
    print(f"  💾 Model size: {model_size_mb:.1f} MB")
    print(f"  🏗️  Architecture: U-Net (features=32)")
    print(f"  📐 Input: (3, 256, 256) → Output: (1, 256, 256)")
    
    # === 2. LOAD TEST DATA ===
    print("\n" + "─" * 80)
    print("  [2/6] LOADING TEST DATA")
    print("─" * 80)
    
    data_root = Path(__file__).parent / '..'
    samples = load_test_samples(
        data_root / 'data/images',
        data_root / 'data/binary_masks',
        max_samples=300
    )
    print(f"  ✅ Loaded {len(samples)} image-mask pairs")
    
    if len(samples) == 0:
        print("  ❌ No test samples found!")
        return
    
    # === 3. RUN INFERENCE & COMPUTE METRICS ===
    print("\n" + "─" * 80)
    print("  [3/6] RUNNING INFERENCE ON TEST SET")
    print("─" * 80)
    
    all_metrics = []
    water_percentages = []
    status_counts = {"Aman": 0, "Siaga": 0, "Waspada": 0, "Bahaya": 0}
    inference_times = []
    
    # Store some samples for visualization
    viz_samples = []
    viz_indices = [0, len(samples)//4, len(samples)//2, 3*len(samples)//4, len(samples)-1]
    
    for i, (img_path, mask_path) in enumerate(samples):
        # Inference
        t0 = time.time()
        image_rgb, pred_mask = predict_single(model, img_path, device)
        inference_times.append(time.time() - t0)
        
        # Load ground truth
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask_norm = gt_mask.astype(np.float32) / 255.0
        
        # Resize pred to match GT
        pred_resized = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
        
        # Metrics
        metrics = calculate_metrics(pred_resized, gt_mask_norm)
        all_metrics.append(metrics)
        
        # Water percentage
        water_pct = np.sum(pred_resized > 0.5) / pred_resized.size * 100
        water_percentages.append(water_pct)
        status, _ = get_flood_status(water_pct)
        status_counts[status] += 1
        
        # Save viz samples
        if i in viz_indices:
            viz_samples.append({
                'image': image_rgb,
                'pred': pred_resized,
                'gt': gt_mask_norm,
                'metrics': metrics,
                'water_pct': water_pct,
                'name': img_path.name
            })
        
        if (i + 1) % 50 == 0 or i == len(samples) - 1:
            print(f"  Progress: {i+1}/{len(samples)} samples processed")
    
    # === 4. AGGREGATE RESULTS ===
    print("\n" + "─" * 80)
    print("  [4/6] TEST RESULTS SUMMARY")
    print("─" * 80)
    
    avg_metrics = {}
    for key in ['accuracy', 'precision', 'recall', 'f1', 'iou', 'dice']:
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    print(f"\n  {'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'─'*55}")
    for key, val in avg_metrics.items():
        print(f"  {key.upper():<15} {val['mean']:>10.4f} {val['std']:>10.4f} {val['min']:>10.4f} {val['max']:>10.4f}")
    
    avg_time = np.mean(inference_times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"\n  ⏱️  Avg inference time: {avg_time*1000:.1f} ms/image")
    print(f"  ⚡ Throughput: {fps:.1f} FPS")
    
    print(f"\n  🌊 Flood Status Distribution ({len(samples)} samples):")
    for status, count in status_counts.items():
        pct = count / len(samples) * 100
        _, icon = get_flood_status({"Aman": 0, "Siaga": 10, "Waspada": 20, "Bahaya": 50}[status])
        bar = "█" * int(pct / 2)
        print(f"  {icon} {status:<10} {count:>4} ({pct:>5.1f}%) {bar}")
    
    print(f"\n  💧 Water Coverage Stats:")
    print(f"     Mean: {np.mean(water_percentages):.2f}%")
    print(f"     Std:  {np.std(water_percentages):.2f}%")
    print(f"     Min:  {np.min(water_percentages):.2f}%")
    print(f"     Max:  {np.max(water_percentages):.2f}%")
    
    # === 5. GENERATE VISUALIZATIONS ===
    print("\n" + "─" * 80)
    print("  [5/6] GENERATING VISUALIZATIONS")
    print("─" * 80)
    
    output_dir = data_root / 'test_results'
    output_dir.mkdir(exist_ok=True)
    
    # --- 5a. Sample Predictions Grid ---
    n_viz = min(5, len(viz_samples))
    fig, axes = plt.subplots(n_viz, 4, figsize=(20, 5 * n_viz))
    fig.suptitle('U-Net Flood Detection - Sample Predictions', fontsize=18, fontweight='bold', y=0.98)
    
    col_titles = ['Original Image', 'Ground Truth', 'Prediction', 'Overlay']
    
    for j, title in enumerate(col_titles):
        if n_viz > 1:
            axes[0, j].set_title(title, fontsize=14, fontweight='bold')
        else:
            axes[j].set_title(title, fontsize=14, fontweight='bold')
    
    for i, sample in enumerate(viz_samples[:n_viz]):
        row = axes[i] if n_viz > 1 else axes
        
        # Original
        row[0].imshow(sample['image'])
        row[0].set_ylabel(f"{sample['name']}\nIoU: {sample['metrics']['iou']:.3f}", fontsize=10)
        row[0].set_xticks([]); row[0].set_yticks([])
        
        # Ground Truth
        row[1].imshow(sample['gt'], cmap='Blues', vmin=0, vmax=1)
        gt_pct = np.sum(sample['gt'] > 0.5) / sample['gt'].size * 100
        row[1].set_xlabel(f"Water: {gt_pct:.1f}%", fontsize=10)
        row[1].set_xticks([]); row[1].set_yticks([])
        
        # Prediction
        row[2].imshow(sample['pred'], cmap='Blues', vmin=0, vmax=1)
        status, icon = get_flood_status(sample['water_pct'])
        row[2].set_xlabel(f"Water: {sample['water_pct']:.1f}% ({status})", fontsize=10)
        row[2].set_xticks([]); row[2].set_yticks([])
        
        # Overlay
        overlay = sample['image'].copy()
        mask_colored = np.zeros_like(overlay)
        binary = (sample['pred'] > 0.5)
        h, w = binary.shape
        img_resized = cv2.resize(sample['image'], (w, h))
        mask_colored[binary] = [0, 120, 255]
        blended = cv2.addWeighted(img_resized, 0.6, mask_colored, 0.4, 0)
        row[3].imshow(blended)
        row[3].set_xticks([]); row[3].set_yticks([])
    
    plt.tight_layout()
    sample_path = output_dir / 'test_sample_predictions.png'
    fig.savefig(str(sample_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Sample predictions → {sample_path.name}")
    
    # --- 5b. Metrics Dashboard ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Test Metrics Dashboard', fontsize=18, fontweight='bold')
    
    # IoU Distribution
    iou_vals = [m['iou'] for m in all_metrics]
    axes[0, 0].hist(iou_vals, bins=30, color='#2196F3', edgecolor='white', alpha=0.8)
    axes[0, 0].axvline(np.mean(iou_vals), color='red', linestyle='--', label=f'Mean: {np.mean(iou_vals):.4f}')
    axes[0, 0].set_title('IoU Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('IoU Score')
    axes[0, 0].legend()
    
    # F1 Distribution
    f1_vals = [m['f1'] for m in all_metrics]
    axes[0, 1].hist(f1_vals, bins=30, color='#4CAF50', edgecolor='white', alpha=0.8)
    axes[0, 1].axvline(np.mean(f1_vals), color='red', linestyle='--', label=f'Mean: {np.mean(f1_vals):.4f}')
    axes[0, 1].set_title('F1 Score Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('F1 Score')
    axes[0, 1].legend()
    
    # Precision vs Recall
    prec_vals = [m['precision'] for m in all_metrics]
    rec_vals = [m['recall'] for m in all_metrics]
    axes[0, 2].scatter(rec_vals, prec_vals, alpha=0.5, s=15, color='#FF9800')
    axes[0, 2].set_title('Precision vs Recall', fontweight='bold')
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].set_xlim(0, 1.05)
    axes[0, 2].set_ylim(0, 1.05)
    
    # Water Percentage Distribution
    axes[1, 0].hist(water_percentages, bins=30, color='#00BCD4', edgecolor='white', alpha=0.8)
    axes[1, 0].axvline(5, color='green', linestyle='--', alpha=0.7, label='Siaga (5%)')
    axes[1, 0].axvline(15, color='orange', linestyle='--', alpha=0.7, label='Waspada (15%)')
    axes[1, 0].axvline(30, color='red', linestyle='--', alpha=0.7, label='Bahaya (30%)')
    axes[1, 0].set_title('Water Coverage Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Water %')
    axes[1, 0].legend(fontsize=8)
    
    # Flood Status Pie Chart
    labels = list(status_counts.keys())
    sizes = list(status_counts.values())
    colors = ['#4CAF50', '#FFEB3B', '#FF9800', '#F44336']
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if non_zero:
        l, s, c = zip(*non_zero)
        axes[1, 1].pie(s, labels=l, colors=c, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Flood Status Classification', fontweight='bold')
    
    # Summary Table
    axes[1, 2].axis('off')
    summary_text = (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  MODEL TEST SUMMARY\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  Test Samples:   {len(samples)}\n"
        f"  Mean IoU:       {np.mean(iou_vals):.4f}\n"
        f"  Mean F1:        {np.mean(f1_vals):.4f}\n"
        f"  Mean Accuracy:  {avg_metrics['accuracy']['mean']:.4f}\n"
        f"  Mean Precision: {avg_metrics['precision']['mean']:.4f}\n"
        f"  Mean Recall:    {avg_metrics['recall']['mean']:.4f}\n"
        f"  Mean Dice:      {avg_metrics['dice']['mean']:.4f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  Inference: {avg_time*1000:.1f}ms ({fps:.1f} FPS)\n"
        f"  Device: {device.upper()}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    metrics_path = output_dir / 'test_metrics_dashboard.png'
    fig.savefig(str(metrics_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Metrics dashboard → {metrics_path.name}")
    
    # --- 5c. Training History ---
    history_path = data_root / 'model/training_history.json'
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Train Loss')
        ax1.plot(epochs, history['val_loss'], 'r-o', markersize=4, label='Val Loss')
        ax1.set_title('Loss Curve')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (BCE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, history['val_iou'], 'g-o', markersize=4, label='Val IoU')
        best_epoch = np.argmax(history['val_iou']) + 1
        best_iou = max(history['val_iou'])
        ax2.axhline(best_iou, color='red', linestyle='--', alpha=0.5)
        ax2.annotate(f'Best: {best_iou:.4f} (Epoch {best_epoch})',
                    xy=(best_epoch, best_iou), fontsize=10, color='red')
        ax2.set_title('Validation IoU')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IoU')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        history_fig_path = output_dir / 'test_training_history.png'
        fig.savefig(str(history_fig_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✅ Training history → {history_fig_path.name}")
    
    # === 6. SAVE JSON REPORT ===
    print("\n" + "─" * 80)
    print("  [6/6] SAVING TEST REPORT")
    print("─" * 80)
    
    report = {
        "test_date": datetime.now().isoformat(),
        "device": device,
        "model": {
            "path": str(model_path),
            "parameters": params,
            "size_mb": round(model_size_mb, 2),
            "architecture": "U-Net (features=32)"
        },
        "test_data": {
            "total_samples": len(samples),
            "source": "data/images + data/binary_masks"
        },
        "metrics": {k: {"mean": round(v['mean'], 6), "std": round(v['std'], 6)} 
                    for k, v in avg_metrics.items()},
        "performance": {
            "avg_inference_ms": round(avg_time * 1000, 2),
            "fps": round(fps, 2)
        },
        "flood_distribution": {k: v for k, v in status_counts.items()},
        "water_coverage": {
            "mean_pct": round(np.mean(water_percentages), 2),
            "std_pct": round(np.std(water_percentages), 2),
            "min_pct": round(np.min(water_percentages), 2),
            "max_pct": round(np.max(water_percentages), 2)
        }
    }
    
    report_path = output_dir / 'test_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  ✅ JSON report → {report_path.name}")
    
    # === FINAL SUMMARY ===
    print("\n" + "=" * 80)
    print("  ✅ MODEL TEST COMPLETE")
    print("=" * 80)
    print(f"  📊 Tested on {len(samples)} samples")
    print(f"  🎯 Mean IoU:       {avg_metrics['iou']['mean']:.4f}")
    print(f"  🎯 Mean F1:        {avg_metrics['f1']['mean']:.4f}")
    print(f"  🎯 Mean Accuracy:  {avg_metrics['accuracy']['mean']:.4f}")
    print(f"  🎯 Mean Precision: {avg_metrics['precision']['mean']:.4f}")
    print(f"  🎯 Mean Recall:    {avg_metrics['recall']['mean']:.4f}")
    print(f"  ⚡ Speed: {fps:.1f} FPS ({avg_time*1000:.1f} ms/image)")
    print(f"\n  📁 Results saved to: {output_dir.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    run_test()
