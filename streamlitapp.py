import cv2
import numpy as np
import av
from ultralytics import YOLO
import mediapipe as mp
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import streamlit as st
import requests
import os

# Function to download model from Google Drive
def download_model():
    url = "https://drive.google.com/uc?id=1fL7so9KFxekf-U7-TgYKDnnE-7W2bVng"  # Replace FILE_ID with your actual file ID
    model_path = "best.pt"

    if not os.path.exists(model_path):
        st.write("Downloading the YOLO model...")
        response = requests.get(url, stream=True)
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        st.write("Model download complete.")

    return model_path

# Load YOLO model dynamically
model_path = download_model()
model = YOLO(model_path).to('cuda')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    ab = np.array([b[0] - a[0], b[1] - a[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    dot_product = np.dot(ab, bc)
    mag_ab = np.linalg.norm(ab)
    mag_bc = np.linalg.norm(bc)
    angle = np.degrees(np.arccos(dot_product / (mag_ab * mag_bc)))
    return angle

# Function to analyze posture
def analyze_posture(landmarks, exercise_type):
    feedback = "Good Form"

    if exercise_type == 'bicep curl':
        angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        if angle < 30 or angle > 160:
            feedback = "Bad Form: Adjust Elbow Angle"

    elif exercise_type == 'push-up':
        angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        if angle < 90 or angle > 160:
            feedback = "Bad Form: Adjust Elbow Position"

    elif exercise_type == 'squat':
        angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        if angle < 80 or angle > 150:
            feedback = "Bad Form: Adjust Knee Angle"

    return feedback

# Streamlit App
st.title("Real-Time Exercise Feedback")
st.write("Use your phone or webcam for real-time exercise feedback.")

# Video transformer class for Streamlit WebRTC
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")  # Convert WebRTC frame to OpenCV format

        # Run YOLO classification
        results = model(image)
        if len(results[0].boxes):
            sorted_boxes = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)
            class_id = int(sorted_boxes[0].cls)
            exercise_type = {0: 'bicep curl', 1: 'push-up', 2: 'squat'}.get(class_id, 'Unknown')
        else:
            exercise_type = 'Unknown'

        # Run MediaPipe pose estimation
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image_rgb)

        feedback = "Unknown"
        if pose_results.pose_landmarks:
            landmarks = [(lm.x * image.shape[1], lm.y * image.shape[0]) for lm in pose_results.pose_landmarks.landmark]
            feedback = analyze_posture(landmarks, exercise_type)
            mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Annotate the frame with YOLO results and feedback
        cv2.putText(image, f"Exercise: {exercise_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        color = (0, 255, 0) if feedback == "Good Form" else (0, 0, 255)
        cv2.putText(image, feedback, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Streamlit WebRTC interface
webrtc_streamer(key="exercise-feedback", video_transformer_factory=VideoTransformer)
