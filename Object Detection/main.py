import os
from src.webcam_detection import run_webcam_detection
from utils.helper import ensure_dir

def setup_directories():
    """
    Ensure all necessary directories exist before running.
    """
    dirs = [
        "models",
        os.path.join("dataset", "images"),
        os.path.join("dataset", "labels"),
        os.path.join("outputs", "detected_images"),
        os.path.join("outputs", "detected_videos")
    ]
    
    for d in dirs:
        ensure_dir(d)

def main():
    setup_directories()
    
    print("=== YOLOv8 Real-Time Object Detection ===")
    print("Starting Webcam Detection...")
    
    # Directly start webcam detection
    run_webcam_detection()

if __name__ == "__main__":
    main()
