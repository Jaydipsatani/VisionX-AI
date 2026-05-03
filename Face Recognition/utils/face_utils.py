import os
import math

def initialize_folders():
    """
    Ensure the required directories exist for the project.
    """
    base_dirs = ["dataset", "models", "outputs"]
    for dir_name in base_dirs:
        os.makedirs(dir_name, exist_ok=True)
    print("Folder structure initialized.")

def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    """
    Evaluates Euclidean distance between face encodings and returns a confidence percentage.
    Lower distance = Higher confidence.
    """
    if face_distance > face_match_threshold:
        range_dist = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range_dist * 2.0)
        return linear_val
    else:
        range_dist = face_match_threshold
        linear_val = 1.0 - (face_distance / (range_dist * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def format_confidence(confidence):
    """
    Returns formatted percentage string.
    """
    return f"{round(confidence * 100, 2)}%"
