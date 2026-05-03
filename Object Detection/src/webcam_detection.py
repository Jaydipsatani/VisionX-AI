import cv2
import os
from src.object_detection import ObjectDetector
from utils.helper import get_video_writer

def run_webcam_detection():
    """
    Run object detection on the live webcam feed and save the output.
    """
    print("Starting webcam detection. Press 'q' to stop.")
    
    # Initialize the detector
    detector = ObjectDetector()
    
    # Open the default webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    # Get webcam properties for video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30 # Default to 30 fps
        
    # Initialize video writer to save the stream
    output_path = os.path.join("outputs", "detected_videos", "webcam_output.mp4")
    out = get_video_writer(output_path, fps, width, height)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            # Perform object detection and formatting
            annotated_frame = detector.detect(frame)
            
            # Write the frame to the output video
            out.write(annotated_frame)
            
            # Display the resulting frame on screen
            cv2.imshow('Webcam Object Detection - YOLOv8', annotated_frame)
            
            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release everything when done
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Webcam detection stopped. Video saved to {output_path}")
