import os
import cv2
import face_recognition
from utils.face_utils import initialize_folders

def register_user(name: str, num_samples: int = 30):
    """
    Captures face images from the webcam and saves them in the dataset folder.
    
    Args:
        name (str): The name of the user to register.
        num_samples (int): The number of images to capture. Default is 30.
    """
    initialize_folders()
    
    # Create directory for the user
    user_dir = os.path.join("dataset", name)
    os.makedirs(user_dir, exist_ok=True)
    
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return
        
    print(f"Starting registration for user: {name}")
    print("Please look at the camera. Capturing will start...")
    
    count = 0
    while count < num_samples:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Handle cases
        if len(face_locations) == 0:
            cv2.putText(frame, "No face detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif len(face_locations) > 1:
            cv2.putText(frame, "Multiple faces detected! Please ensure only one face is visible.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # We have exactly one face
            top, right, bottom, left = face_locations[0]
            
            # Crop the face from the frame using array slicing
            face_image = frame[top:bottom, left:right]
            
            # Save the image
            file_path = os.path.join(user_dir, f"{count}.jpg")
            try:
                cv2.imwrite(file_path, face_image)
                count += 1
                
                # Draw a box around the face and display count
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"Captured: {count}/{num_samples}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error saving image: {e}")
                
        # Display the resulting image
        cv2.imshow('Registration', frame)
        
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    
    if count == num_samples:
        print(f"Registration successful for {name}.")
    else:
        print(f"Registration stopped. Captured {count}/{num_samples} images.")

if __name__ == "__main__":
    name = input("Enter user name to register: ")
    if name.strip():
        register_user(name.strip())
    else:
        print("Invalid name. Exiting.")
