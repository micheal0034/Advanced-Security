import streamlit as st
import cv2
import numpy as np
from thermal_analyzer import ThermalAnalyzer
from movement_analyzer import MovementAnalyzer
from face_analyzer import FaceAnalyzer
from imposter_detector import ImposterDetector
from config import SecurityConfig
from PIL import Image
import time
from ultralytics import YOLO  # Import YOLOv8

# Initialize Security Config
config = SecurityConfig()

# Initialize Analyzers
thermal_analyzer = ThermalAnalyzer(config)
movement_analyzer = MovementAnalyzer(config)
face_analyzer = FaceAnalyzer(config)
imposter_detector = ImposterDetector(thermal_analyzer, face_analyzer, movement_analyzer)

# Initialize YOLOv8 Model (using a smaller model like yolov8n for fast performance)
yolo_model = YOLO("yolov8n.pt")  # Replace with your YOLO model path

# Streamlit App
st.set_page_config(page_title="Imposter Detection System", layout="wide")
st.title("Imposter Detection System")

# Password Authentication
password = st.text_input("Enter Password", type="password")  # Input field for password
hardcoded_password = "admin"  # Hardcoded password

if password != hardcoded_password:
    st.error("Invalid password. Please try again.")
else:
    # If password matches, show the main app content

    # Sidebar Configuration
    st.sidebar.header("Camera Settings")
    camera_url = st.sidebar.text_input("Enter Camera URL", "")
    frame_width = st.sidebar.slider("Frame Width", 320, 1280, config.FRAME_WIDTH)
    frame_height = st.sidebar.slider("Frame Height", 240, 720, config.FRAME_HEIGHT)
    detection_interval = st.sidebar.slider("Detection Interval (s)", 0.1, 2.0, config.DETECTION_INTERVAL)

    # Upload option for testing
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "png", "jpeg"])

    # Start Camera
    run_camera = st.button("Start Camera")

    # Display Results
    st.subheader("Detection Results")
    results_placeholder = st.empty()

    def get_detection_box(frame):
        """
        Uses YOLOv8 to detect a person and return the bounding box coordinates.
        """
        # Perform inference on the frame using YOLOv8 model
        results = yolo_model(frame)  # Perform object detection
        detection_box = None
        
        # Loop through results to find the first person (class 0)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if int(box.cls) == 0:  # Class 0 represents person
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detection_box = [int(x1), int(y1), int(x2), int(y2)]
                    break  # We take the first detected person

        return detection_box

    if run_camera or uploaded_file:
        if run_camera:
            if not camera_url:
                st.error("Please provide a valid Camera URL.")
            else:
                # Initialize video stream
                cap = cv2.VideoCapture(camera_url)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

                if not cap.isOpened():
                    st.error("Failed to connect to the camera.")
                else:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to capture frame from the camera.")
                            break

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        detection_box = get_detection_box(frame_rgb)

                        if detection_box:
                            # Run analysis
                            is_imposter, total_score = imposter_detector.detect_imposter(frame_rgb, detection_box)

                            # Draw detection box
                            x1, y1, x2, y2 = detection_box
                            frame_rgb = cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

                            # Display Results
                            results_placeholder.write(f"Imposter Detected: {is_imposter}, Confidence Score: {total_score:.2f}")
                            st.image(frame_rgb, channels="RGB")

                        else:
                            st.write("No person detected.")

                        # Pause for the detection interval
                        time.sleep(detection_interval)

                    cap.release()

        if uploaded_file:
            # Process uploaded image
            image = Image.open(uploaded_file)
            frame_rgb = np.array(image)

            # Get detection box for uploaded image
            detection_box = get_detection_box(frame_rgb)

            if detection_box:
                # Run analysis
                is_imposter, total_score = imposter_detector.detect_imposter(frame_rgb, detection_box)

                # Draw detection box
                x1, y1, x2, y2 = detection_box
                frame_rgb = cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Display Results
                st.write(f"Imposter Detected: {is_imposter}, Confidence Score: {total_score:.2f}")
                st.image(frame_rgb, caption="Uploaded Image", channels="RGB")
            else:
                st.write("No person detected in the uploaded image.")
