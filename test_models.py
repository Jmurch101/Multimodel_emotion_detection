#!/usr/bin/env python3
"""
Test script to isolate model loading issues on macOS
"""
import os
# Set environment variables before any imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import cv2
import numpy as np

def test_opencv():
    """Test basic OpenCV functionality"""
    print("Testing OpenCV...")
    try:
        # Test basic image loading
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("✓ OpenCV basic operations work")
        return True
    except Exception as e:
        print(f"✗ OpenCV failed: {e}")
        return False

def test_person_detection():
    """Test YOLO person detection"""
    print("Testing YOLO person detection...")
    try:
        from detectors import PersonDetector
        detector = PersonDetector()
        # Test on a small dummy image
        dummy_img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        results = detector.detect(dummy_img)
        print(f"✓ YOLO loaded, detected {len(results)} persons in test image")
        return True
    except Exception as e:
        print(f"✗ YOLO failed: {e}")
        return False

def test_face_analysis():
    """Test face analysis (OpenCV Haar cascades, no TensorFlow)"""
    print("Testing face detection (OpenCV Haar cascades)...")
    try:
        from detectors.face_analyzer_opencv import OpenCVFaceAnalyzer
        analyzer = OpenCVFaceAnalyzer()
        # Test on a small dummy image
        dummy_img = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        results = analyzer.analyze(dummy_img, [])
        print(f"✓ Face detection loaded, analyzed test image")
        return True
    except Exception as e:
        print(f"✗ Face detection failed: {e}")
        return False

def test_deepface_analysis():
    """Test DeepFace analysis separately"""
    print("Testing DeepFace analysis (may fail on macOS)...")
    try:
        from detectors.face_analyzer import FaceAnalyzer
        analyzer = FaceAnalyzer()
        # Test on a small dummy image
        dummy_img = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        results = analyzer.analyze(dummy_img, [])
        print(f"✓ DeepFace loaded, analyzed test image")
        return True
    except Exception as e:
        print(f"✗ DeepFace failed: {e}")
        print("  → This is expected on macOS. Use simple face detection instead.")
        return False

def main():
    print("=== Model Compatibility Test ===\n")

    opencv_ok = test_opencv()
    print()

    yolo_ok = test_person_detection()
    print()

    face_ok = test_face_analysis()
    print()

    # Skip DeepFace test since we're using OpenCV-based age/emotion now
    deepface_ok = True  # Not tested, using OpenCV alternative
    print("DeepFace test skipped (using OpenCV-based age/emotion estimation)")
    print()

    print("=== Summary ===")
    print(f"OpenCV: {'✓' if opencv_ok else '✗'}")
    print(f"YOLOv8: {'✓' if yolo_ok else '✗'}")
    print(f"Face Detection + Age/Emotion (OpenCV): {'✓' if face_ok else '✗'}")
    print(f"DeepFace: Skipped (using OpenCV alternative)")

    if not all([opencv_ok, yolo_ok, face_ok]):
        print("\n❌ Core models failed. Check error messages above.")
        return 1
    else:
        print("\n✅ All models loaded successfully!")
        print("   → Age/emotion estimation uses OpenCV-based heuristics")
        return 0

if __name__ == "__main__":
    sys.exit(main())
