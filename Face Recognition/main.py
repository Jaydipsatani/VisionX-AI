import os
import cv2
import pickle
import numpy as np
import face_recognition

from utils.face_utils import initialize_folders, face_distance_to_conf, format_confidence

def start_face_recognition():
    """
    Opens webcam, detects faces, compares with known encodings, and displays
    results in real-time.
    """
    initialize_folders()
    
    model_path = os.path.join("Face Recognition", "models", "encodings.pkl")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        return
        
    print("Loading encodings...")
    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    known_encodings = data["encodings"]
    known_names = data["names"]
    
    if len(known_encodings) == 0:
        print("Model contains no valid encodings. Please register users and re-train.")
        return
        
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return
        
    print("Starting webcam... Press 'q' to quit.")
    
    while True:
        # Grab a single frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert the image from BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all the faces and face encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        face_confidences = []
        
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            confidence = 0.0
            
            # Use the known face with the smallest distance to the new face
            if known_encodings:
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    dist = face_distances[best_match_index]
                    confidence = face_distance_to_conf(dist)
                    
            face_names.append(name)
            face_confidences.append(confidence)
            
        # Display the results
        for (top, right, bottom, left), name, conf in zip(face_locations, face_names, face_confidences):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Choose color based on whether it's known
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            
            # Prepare label string
            if name != "Unknown":
                label = f"{name} {format_confidence(conf)}"
            else:
                label = "Unknown"
                
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
        # Display the resulting image
        cv2.imshow('Face Recognition', frame)
        
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_face_recognition()
