import os
import cv2
import pickle
import face_recognition
from utils.face_utils import initialize_folders

def train_model():
    """
    Reads images from dataset directory, extracts face encodings,
    and saves them to a pickle file in the models directory.
    """
    initialize_folders()
    
    dataset_dir = "dataset"
    model_path = os.path.join("models", "encodings.pkl")
    
    known_encodings = []
    known_names = []
    
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        print("Dataset directory is empty. Please register users first.")
        return
        
    print("Starting training process...")
    
    # Iterate through each user directory in the dataset
    for name in os.listdir(dataset_dir):
        user_dir = os.path.join(dataset_dir, name)
        
        # Check if it's a directory
        if not os.path.isdir(user_dir):
            continue
            
        print(f"Processing images for user: {name}")
        
        # Iterate through each image file
        for filename in os.listdir(user_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(user_dir, filename)
                
                # Load the image
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                        
                    # Convert to RGB
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Get face bounding boxes
                    boxes = face_recognition.face_locations(rgb_image)
                    
                    # Compute encodings for faces
                    encodings = face_recognition.face_encodings(rgb_image, boxes)
                    
                    # Suppose each image has exactly one face
                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_names.append(name)
                    else:
                        print(f"Warning: No face found in {image_path}. Skipping.")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    
    print(f"Total encodings generated: {len(known_encodings)}")
    
    # Dump encodings to a pickle file
    print(f"Saving encodings to {model_path}...")
    data = {"encodings": known_encodings, "names": known_names}
    
    with open(model_path, "wb") as f:
        pickle.dump(data, f)
        
    print("Training complete!")

if __name__ == "__main__":
    train_model()
