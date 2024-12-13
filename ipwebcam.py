import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# Load YOLO model
model = YOLO('C:/Users/narji/Desktop/best/best.pt').to('cuda')

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
st.sidebar.title("Settings")

# Select video source
video_source = st.sidebar.selectbox("Select Video Source", ("DroidCam USB", "Webcam", "Upload Video"))

# Initialize video capture based on the selected source
if video_source == "DroidCam USB":
    st.sidebar.write("Ensure DroidCam is running and connected via USB.")
    cap = cv2.VideoCapture(1)  # Use Camera Index 1 for DroidCam USB
elif video_source == "Webcam":
    cap = cv2.VideoCapture(0)  # Use the default webcam
elif video_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi"])
    if uploaded_file:
        temp_file = "temp_video.mp4"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_file)

if st.sidebar.button("Start"):
    if 'cap' in locals() and cap and cap.isOpened():
        st_frame = st.empty()  # Placeholder for video frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO classification
            results = model(frame)
            if len(results[0].boxes):
                sorted_boxes = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)
                class_id = int(sorted_boxes[0].cls)
                exercise_type = {0: 'bicep curl', 1: 'push-up', 2: 'squat'}.get(class_id, 'Unknown')
            else:
                exercise_type = 'Unknown'

            # Run MediaPipe pose estimation
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(image_rgb)

            feedback = "Unknown"
            if pose_results.pose_landmarks:
                landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in pose_results.pose_landmarks.landmark]
                feedback = analyze_posture(landmarks, exercise_type)
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display exercise type and feedback
            cv2.putText(frame, f"Exercise: {exercise_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            color = (0, 255, 0) if feedback == "Good Form" else (0, 0, 255)
            cv2.putText(frame, feedback, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Stream the frame to Streamlit
            st_frame.image(frame, channels="BGR", use_column_width=True)

        cap.release()
    else:
        st.error("No video source selected or invalid file!")
