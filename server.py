from flask import Flask
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app)

# Virtual Mouse
@app.route('/run-virtual-mouse')
def run_virtual_mouse():
    subprocess.Popen(["python", "AI Virtual Mouse/main.py"])
    return {"message": "Virtual Mouse Started"}

#Face Recognition
@app.route('/run-face-recognition')
def run_face_recognition():
    subprocess.Popen(["python", "Face Recognition/main.py"])
    return {"message": "Face Recognition Started"}

# Object Detection
@app.route('/run-object-detection')
def run_object_detection():
    subprocess.Popen(["python", "Object-Detection-Project/main.py"])
    return {"message": "Object Detection Started"}

if __name__ == "__main__":
    app.run(port=5000)