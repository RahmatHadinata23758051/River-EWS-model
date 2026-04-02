#!/usr/bin/env python3
"""
QUICK INFERENCE SCRIPT
Fast water detection on images/videos with the trained U-Net model
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import argparse

# Import model
import importlib.util
spec = importlib.util.spec_from_file_location("unet_model", Path(__file__).parent / "04_model_unet_architecture.py")
unet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet_module)
create_model = unet_module.create_model


class FloodDetector:
    """Simple flood detection interface"""
    
    def __init__(self, model_path='checkpoints/best_model.pth', device=None, threshold=0.5):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.image_size = 256
        
        # Load model
        self.model = create_model(device=self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    def detect(self, frame):
        """
        Detect water in frame
        
        Args:
            frame: cv2 image (BGR)
            
        Returns:
            dict with keys: water_pct, status, mask, binary_mask
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        resized = cv2.resize(rgb, (self.image_size, self.image_size))
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(tensor)
            output_np = output.cpu().numpy()[0, 0]
        
        # Resize back
        mask = cv2.resize(output_np, (w, h), interpolation=cv2.INTER_LINEAR)
        binary_mask = (mask > self.threshold).astype(np.uint8) * 255
        
        # Calculate water percentage
        water_pct = np.sum(mask > self.threshold) / mask.size * 100
        
        # Status
        if water_pct < 5:
            status = "Aman"
        elif water_pct < 15:
            status = "Siaga"
        elif water_pct < 30:
            status = "Waspada"
        else:
            status = "Bahaya"
        
        return {
            'water_pct': water_pct,
            'status': status,
            'mask': mask,
            'binary_mask': binary_mask
        }


def process_image(image_path, detector, save_overlay=False):
    """Process single image"""
    print(f"Processing: {image_path}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"✗ Cannot open image: {image_path}")
        return
    
    result = detector.detect(image)
    
    print(f"  Water: {result['water_pct']:.1f}%")
    print(f"  Status: {result['status']}")
    
    if save_overlay:
        output_path = Path(image_path).parent / f"{Path(image_path).stem}_water.jpg"
        
        # Create overlay
        overlay = image.copy()
        water_pixels = result['binary_mask'] == 255
        overlay[water_pixels] = [255, 255, 0]  # Cyan
        overlay_blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        # Add text
        text = f"Water: {result['water_pct']:.1f}% | {result['status']}"
        cv2.putText(overlay_blended, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite(str(output_path), overlay_blended)
        print(f"  ✓ Overlay saved: {output_path}")


def process_video(video_path, detector, save_overlay=True):
    """Process video file"""
    print(f"Processing: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"✗ Cannot open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output
    if save_overlay:
        output_path = Path(video_path).parent / f"{Path(video_path).stem}_water.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    else:
        out = None
    
    # Process frames
    frame_idx = 0
    water_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.detect(frame)
        water_list.append(result['water_pct'])
        
        if out:
            overlay = frame.copy()
            water_pixels = result['binary_mask'] == 255
            overlay[water_pixels] = [255, 255, 0]
            overlay_blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            text = f"Water: {result['water_pct']:.1f}% | {result['status']}"
            cv2.putText(overlay_blended, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out.write(overlay_blended)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Progress: {frame_idx}/{total} frames ({frame_idx/total*100:.0f}%)")
    
    cap.release()
    if out:
        out.release()
    
    if water_list:
        print(f"  Avg water: {np.mean(water_list):.1f}%")
        print(f"  Max water: {np.max(water_list):.1f}%")
    
    if save_overlay:
        print(f"  ✓ Output video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Water detection inference')
    parser.add_argument('input', help='Image or video file')
    parser.add_argument('--model', default='checkpoints/best_model.pth', help='Model path')
    parser.add_argument('--device', choices=['cuda', 'cpu'], help='Device (auto-detect if not specified)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Water detection threshold')
    parser.add_argument('--no-overlay', action='store_true', help='Do not save overlay output')
    
    args = parser.parse_args()
    
    # Check input exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ File not found: {args.input}")
        sys.exit(1)
    
    # Load detector
    print(f"\n{'='*60}")
    print(f"FLOOD WATER DETECTION")
    print(f"{'='*60}\n")
    
    detector = FloodDetector(model_path=args.model, device=args.device, threshold=args.threshold)
    print(f"✓ Model loaded (Val IoU: 0.9408)")
    print(f"✓ Device: {detector.device}\n")
    
    # Process input
    if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        process_video(input_path, detector, save_overlay=not args.no_overlay)
    else:
        process_image(input_path, detector, save_overlay=not args.no_overlay)
    
    print(f"\n✓ Done!")


if __name__ == "__main__":
    main()
