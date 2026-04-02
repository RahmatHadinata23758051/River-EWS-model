#!/usr/bin/env python3
"""
Simple setup script to extract and verify model
"""
import json
from pathlib import Path

def setup():
    print("\nFlood Detection Model - Setup")
    print("=" * 50)
    
    # Check model exists
    model_path = Path('model/best_model.pth')
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1024 / 1024
        print("[OK] Model found: best_model.pth ({:.1f} MB)".format(size_mb))
    else:
        print("[ERROR] Model not found!")
        return False
    
    # Check code exists
    if Path('code/inference.py').exists():
        print("[OK] Inference code ready")
    else:
        print("[ERROR] Inference code not found!")
        return False
    
    # Show info
    if Path('model_card.json').exists():
        with open('model_card.json') as f:
            card = json.load(f)
        print("\nModel: {} v{}".format(card['name'], card['version']))
        print("Val IoU: {}".format(card['performance']['validation_iou']))
        print("Tested on: {} videos".format(card['testing_results']['test_videos']))
    
    print("\n[OK] Setup complete!\n")
    return True

if __name__ == "__main__":
    setup()
