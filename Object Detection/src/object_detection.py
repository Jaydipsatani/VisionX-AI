import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="models/yolov8n.pt"):
        """
        Initialize the YOLOv8 object detector.
        If the model isn't downloaded, ultralytics will download it automatically.
        """
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Detect objects in a given frame.
        Draw bounding boxes and labels automatically using ultralytics visualizer.
        """
        # Run YOLOv8 inference on the frame
        results = self.model(frame, verbose=False)
        
        # Visualize the results on the frame
        # plot() draws the bounding boxes and labels.
        annotated_frame = results[0].plot()
        
        return annotated_frame
