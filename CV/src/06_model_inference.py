"""
Inference Pipeline
Testing trained model on new images and videos
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import importlib.util

# Import U-Net architecture dynamically
spec = importlib.util.spec_from_file_location(
    "unet_model", 
    Path(__file__).parent / "04_model_unet_architecture.py"
)
unet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet_module)
create_model = unet_module.create_model


class FloodDetector:
    """Flood detection inference engine"""
    
    def __init__(self, model_path, device='cuda', threshold=0.5):
        self.device = device
        self.threshold = threshold
        self.image_size = 256
        
        # Load model
        self.model = create_model(device=device)
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"✓ Model loaded from: {model_path}")
    
    def process_image(self, image_path, return_visualization=False):
        """
        Process single image and return water detection mask
        
        Returns:
            - segmentation_mask: (H, W) binary mask (0-255)
            - flood_detected: boolean
            - confidence: float (0-1)
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_h, original_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        image_resized = cv2.resize(image_rgb, (self.image_size, self.image_size))
        image_tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            output_np = output.cpu().numpy()[0, 0]  # (256, 256)
        
        # Resize back to original
        mask_original = cv2.resize(output_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        # Binary mask
        binary_mask = (mask_original > self.threshold).astype(np.uint8) * 255
        
        # Calculate confidence
        water_percentage = np.sum(mask_original > self.threshold) / mask_original.size
        flood_detected = water_percentage > 0.05  # At least 5% water
        
        result = {
            'segmentation_mask': mask_original,  # (0-1) probability
            'binary_mask': binary_mask,  # (0-255) binary
            'flood_detected': flood_detected,
            'water_percentage': water_percentage * 100,
            'flood_status': self._get_flood_status(water_percentage),
        }
        
        if return_visualization:
            result['visualization'] = self._create_visualization(image_rgb, mask_original, binary_mask)
        
        return result
    
    def _get_flood_status(self, water_percentage):
        """Classify flood status based on water percentage"""
        if water_percentage < 0.05:
            return "Aman"  # Safe
        elif water_percentage < 0.15:
            return "Siaga"  # Alert
        elif water_percentage < 0.3:
            return "Waspada"  # Warning
        else:
            return "Bahaya"  # Danger
    
    def _create_visualization(self, image_rgb, mask_probability, binary_mask):
        """Create visualization overlay"""
        # Color overlay
        colored_mask = np.zeros_like(image_rgb)
        colored_mask[binary_mask == 255] = [0, 100, 255]  # Red for water
        
        # Blend
        alpha = 0.5
        overlay = cv2.addWeighted(image_rgb, 1-alpha, colored_mask, alpha, 0)
        
        return overlay
    
    def process_video(self, video_path, output_path=None):
        """Process video and extract water detection info"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {frame_width}x{frame_height} @ {fps:.2f} FPS, {total_frames} frames")
        
        # Output video writer (if specified)
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        frame_results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Preprocess frame
            original_h, original_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.image_size, self.image_size))
            frame_tensor = torch.from_numpy(frame_resized.astype(np.float32) / 255.0)
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(frame_tensor)
                output_np = output.cpu().numpy()[0, 0]
            
            # Resize back
            mask = cv2.resize(output_np, (original_w, original_h))
            binary_mask = (mask > self.threshold).astype(np.uint8) * 255
            water_percentage = np.sum(mask > self.threshold) / mask.size * 100
            
            # Record result
            frame_results.append({
                'frame': frame_count,
                'water_percentage': water_percentage,
                'flood_detected': water_percentage > 5,
                'status': self._get_flood_status(water_percentage / 100)
            })
            
            # Create output frame with overlay
            if out is not None:
                colored_mask = np.zeros_like(frame)
                colored_mask[binary_mask == 255] = [0, 100, 255]  # Red for water
                output_frame = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
                
                # Add text
                text = f"Water: {water_percentage:.1f}% | Status: {self._get_flood_status(water_percentage / 100)}"
                cv2.putText(output_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                out.write(output_frame)
            
            if frame_count % 30 == 0:
                print(f"  Frame {frame_count}/{total_frames}")
        
        cap.release()
        if out is not None:
            out.release()
            print(f"Output video saved to: {output_path}")
        
        return frame_results


def test_inference():
    """Test inference pipeline"""
    
    print("="*80)
    print("FLOOD DETECTION INFERENCE TEST")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Check for trained model
    model_path = Path('../model/best_model.pth')
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please run 05_model_train.py first to train the model")
        return
    
    # Initialize detector
    detector = FloodDetector(str(model_path), device=device)
    
    # Test on sample image
    images_dir = Path('../data/images')
    sample_images = list(images_dir.rglob('*.jpg'))[:3]
    
    if sample_images:
        print(f"\n[1] TESTING ON {len(sample_images)} SAMPLE IMAGES")
        for img_path in sample_images:
            print(f"\nProcessing: {img_path.name}")
            
            result = detector.process_image(str(img_path), return_visualization=True)
            print(f"  Water: {result['water_percentage']:.2f}%")
            print(f"  Flood Detected: {result['flood_detected']}")
            print(f"  Status: {result['flood_status']}")
    
    # Test on video
    video_dir = Path('../data/video')
    sample_videos = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
    
    if sample_videos:
        print(f"\n[2] TESTING ON VIDEO")
        video_path = sample_videos[0]
        output_path = Path('output_video_with_detection.mp4')
        
        print(f"Processing: {video_path.name}")
        frame_results = detector.process_video(str(video_path), str(output_path))
        
        print(f"\nVideo Analysis Summary:")
        flood_frames = sum(1 for f in frame_results if f['flood_detected'])
        print(f"  Total frames: {len(frame_results)}")
        print(f"  Flood detected frames: {flood_frames}")
        print(f"  Average water percentage: {np.mean([f['water_percentage'] for f in frame_results]):.2f}%")


if __name__ == "__main__":
    test_inference()
