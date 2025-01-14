# ExercisePoseCorrection
This project implements a real-time fitness tracking and posture correction system using computer vision and deep learning techniques. It focuses on detecting and classifying three key exercises—push-ups, squats, and bicep curls—while providing real-time feedback on exercise form.
🚀 Features:
  •Exercise Classification: Utilizes YOLOv8 for classifying push-ups, squats, and bicep curls.
  •Pose Estimation: Leverages MediaPipe Pose to track skeletal landmarks and analyze exercise form.
  •Real-Time Feedback: Offers immediate suggestions and corrections to improve workout posture and reduce the risk of injuries.
  •Multi-Input Support:
        •Webcam Integration
        •DroidCam USB for smartphone usage
        •Pre-recorded video uploads
  •User-Friendly Deployment: Built using Streamlit for an interactive and intuitive interface.

🛠️ Methodology
1. Data Collection
    •Videos of exercises recorded at 60 FPS.
    •Frames extracted and annotated using Roboflow.
    •Data augmentation for diversity (brightness, rotation, flipping).
    •Dataset split into:
        •Training Set (70%)
        •Validation Set (20%)
        •Test Set (10%)

2. Model Training
    •Trained using YOLOv8 architecture for robust and fast exercise detection.
    •Optimized on GPU for efficiency.
    •Metrics:
        •Precision: 99%
        •Recall: 89%
        •mAP50-95: 92.7%

3. Pose Estimation & Form Analysis
    •MediaPipe Pose identifies body landmarks.
    •Specific posture criteria ensure correct exercise execution:
        •Push-Ups: Detect up/down phases and evaluate back alignment.
        •Squats: Analyze knee alignment, shoulder-to-knee posture, and back angle.
        •Bicep Curls: Assess elbow movement and shoulder stability.

4. Deployment
    •Streamlit app allows real-time feedback and video analysis.
    •Three modes of input:
        •Webcam for live feedback.
        •DroidCam USB to use a smartphone as a webcam.
        •Video Uploads for offline analysis.

📊 Results

    •High accuracy in detecting and classifying exercises.
    •Comprehensive feedback on form:
        •Real-time corrections using skeletal landmarks.
        •Detailed metrics like joint angles and motion tracking.
    •Loss curves and precision-recall graphs demonstrate strong model performance.

🧑‍💻 Contributors

    Mohammad Ali Haider 
    Syed Afraz 
    Mufti Muqaram Majid Farooqi 
