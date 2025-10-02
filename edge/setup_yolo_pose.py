#!/usr/bin/env python3
"""
Setup script for YOLO11 pose migration.
Downloads the specified YOLO pose model and verifies installation.
"""
import os
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("YOLO11 Pose Setup Script")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Error: Python 3.10+ required")
        return False
    
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check ultralytics installation
    try:
        import ultralytics
        print(f"✅ Ultralytics version: {ultralytics.__version__}")
        
        # Check if version is >= 8.3.0
        version_parts = ultralytics.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major < 8 or (major == 8 and minor < 3):
            print(f"⚠️  Warning: Ultralytics {ultralytics.__version__} detected. Version 8.3.0+ recommended.")
            print("   Run: pip install --upgrade ultralytics")
    except ImportError:
        print("❌ Ultralytics not installed")
        print("   Run: pip install -r requirements.txt")
        return False
    
    # Check if mediapipe is still installed (should be removed)
    try:
        import mediapipe
        print("⚠️  Warning: MediaPipe is still installed (no longer needed)")
        print("   Run: pip uninstall mediapipe")
    except ImportError:
        print("✅ MediaPipe removed")
    
    # Model selection
    print("\n" + "=" * 60)
    print("Select YOLO11 Pose Model:")
    print("=" * 60)
    print("1. yolo11s-pose.pt  - Small   (Fast,  58.9 mAP, 90ms CPU)")
    print("2. yolo11m-pose.pt  - Medium  (Balanced, 64.9 mAP, 187ms CPU) [RECOMMENDED]")
    print("3. yolo11l-pose.pt  - Large   (Accurate, 66.1 mAP, 248ms CPU)")
    print("4. yolo11x-pose.pt  - XLarge  (Best, 69.5 mAP, 488ms CPU)")
    
    choice = input("\nEnter choice (1-4) [default: 2]: ").strip() or "2"
    
    models = {
        "1": "yolo11s-pose.pt",
        "2": "yolo11m-pose.pt",
        "3": "yolo11l-pose.pt",
        "4": "yolo11x-pose.pt"
    }
    
    if choice not in models:
        print("❌ Invalid choice")
        return False
    
    model_name = models[choice]
    print(f"\n✅ Selected: {model_name}")
    
    # Check if model already exists
    model_path = Path(__file__).parent / model_name
    
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
    else:
        print(f"\nℹ️  Model will be downloaded automatically on first run")
        print(f"   Or manually download from:")
        print(f"   https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}")
    
    # Update config file
    print("\n" + "=" * 60)
    print("Configuration Update")
    print("=" * 60)
    
    config_path = Path(__file__).parent / "config.py"
    
    if config_path.exists():
        # Read current config
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Check current setting
        if 'yolo_pose_model_path' in config_content:
            import re
            match = re.search(r'yolo_pose_model_path:\s*str\s*=\s*["\']([^"\']+)["\']', config_content)
            if match:
                current_model = match.group(1)
                print(f"Current model in config: {current_model}")
                
                if current_model != model_name:
                    update = input(f"Update to {model_name}? (y/n) [default: y]: ").strip().lower()
                    if update != 'n':
                        # Update config
                        new_content = re.sub(
                            r'yolo_pose_model_path:\s*str\s*=\s*["\'][^"\']+["\']',
                            f'yolo_pose_model_path: str = "{model_name}"',
                            config_content
                        )
                        
                        with open(config_path, 'w') as f:
                            f.write(new_content)
                        
                        print(f"✅ Config updated to use {model_name}")
                else:
                    print("✅ Config already set correctly")
        else:
            print("⚠️  Warning: yolo_pose_model_path not found in config.py")
    else:
        print("❌ config.py not found")
        return False
    
    # Test YOLO pose model
    print("\n" + "=" * 60)
    print("Testing YOLO Pose Model")
    print("=" * 60)
    
    test = input("Run quick test? (y/n) [default: y]: ").strip().lower()
    
    if test != 'n':
        try:
            from ultralytics import YOLO
            import numpy as np
            
            print(f"Loading {model_name}...")
            model = YOLO(model_name)
            
            print("Creating test image...")
            test_image = np.zeros((640, 640, 3), dtype=np.uint8)
            
            print("Running inference...")
            results = model(test_image, verbose=False)
            
            print("✅ Model loaded and inference successful!")
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'keypoints'):
                    print("✅ Keypoints attribute detected")
                else:
                    print("⚠️  Warning: No keypoints attribute")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    # Summary
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"✅ Model: {model_name}")
    print("✅ Configuration updated")
    print("✅ Dependencies verified")
    print("\nNext steps:")
    print("1. Run: python main.py")
    print("2. Verify pose detection works")
    print("3. Check tap gesture detection")
    print("\nFor more info, see: YOLO_POSE_MIGRATION.md")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
