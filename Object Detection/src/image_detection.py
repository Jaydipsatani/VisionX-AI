import cv2
import os
from src.object_detection import ObjectDetector
from utils.helper import save_image

def run_image_detection(image_filename):
    """
    Run object detection on a specific image and save the output.
    """
    input_path = os.path.join("dataset", "images", image_filename)
    
    if not os.path.exists(input_path):
        print(f"Error: Image {input_path} not found.")
        print(f"Please ensure it is placed in 'dataset/images/'.")
        return
        
    print(f"Running detection on {input_path}...")
    
    # Initialize the detector
    detector = ObjectDetector()
    
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image {input_path}.")
        return
        
    # Perform object detection, bounding box drawing is done inside
    annotated_img = detector.detect(img)
    
    # Save the output image
    output_filename = f"detected_{image_filename}"
    output_path = os.path.join("outputs", "detected_images", output_filename)
    save_image(annotated_img, output_path)
    
    # Optionally display the image
    cv2.imshow('Image Object Detection', annotated_img)
    print("Press any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
