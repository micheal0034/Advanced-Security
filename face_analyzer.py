import tensorflow as tf
import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

class FaceAnalyzer:
    """Analyzes faces for concealment and suspicious features"""
    def __init__(self, config):
        self.config = config
        self.face_detector = self._load_face_detector()
        
    def _load_face_detector(self):
        """Load face detection model"""
        try:
            interpreter = tf.lite.Interpreter(model_path=self.config.FACE_MODEL_PATH)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            logger.error(f"Error loading face detector: {str(e)}")
            raise
            
    def analyze_face(self, frame):
        """Detect face concealment and suspicious features"""
        try:
            # Detect faces
            faces = self._detect_faces(frame)
            if not faces:
                return True, 1.0  # Consider completely hidden face suspicious
            
            # Analyze each detected face
            max_concealment_score = 0.0
            for face in faces:
                concealment_score = self._analyze_concealment(frame, face)
                max_concealment_score = max(max_concealment_score, concealment_score)
            
            is_concealed = max_concealment_score > self.config.FACE_DETECTION_THRESHOLD
            return is_concealed, max_concealment_score
            
        except Exception as e:
            logger.error(f"Error in face analysis: {str(e)}")
            return False, 0.0
    
    # def analyze_face(self, frame):
    #     """Detect face concealment and suspicious features"""
    #     try:
    #         # Detect faces
    #         faces = self._detect_faces(frame)
    #         if len(faces) == 0:  # Changed from 'if not faces:'
    #             return True, 1.0  # Consider completely hidden face suspicious
            
    #         # Analyze each detected face
    #         max_concealment_score = 0.0
    #         for face in faces:
    #             concealment_score = self._analyze_concealment(frame, face)
    #             max_concealment_score = max(max_concealment_score, concealment_score)
            
    #         is_concealed = max_concealment_score > self.config.FACE_DETECTION_THRESHOLD
    #         return is_concealed, max_concealment_score
            
    #     except Exception as e:
    #         logger.error(f"Error in face analysis: {str(e)}")
    #         return False, 0.0

    def _detect_faces(self, frame):
        """Detect faces in frame"""
        # Preprocess frame for face detection
        input_details = self.face_detector.get_input_details()
        output_details = self.face_detector.get_output_details()
        
        # Resize and normalize frame
        preprocessed = cv2.resize(frame, (128, 128))
        preprocessed = (preprocessed / 255.0).astype(np.float32)
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        # Run detection
        self.face_detector.set_tensor(input_details[0]['index'], preprocessed)
        self.face_detector.invoke()
        
        # Get detection results
        # boxes = self.face_detector.get_tensor(output_details[0]['index'])
        # return boxes[boxes[:, 4] > self.config.FACE_DETECTION_THRESHOLD]
        # Get detection results and handle dimensions
        boxes = self.face_detector.get_tensor(output_details[0]['index'])[0]  # Get first batch
        # print(boxes)
        scores = self.face_detector.get_tensor(output_details[1]['index'])[0]  # Get first batch
        scores = scores.flatten()
         # Stable sigmoid implementation to avoid overflow
        scores = np.clip(scores, -88.0, 88.0)  # clip to safe range for exp
        scores = 1 / (1 + np.exp(-scores))
        # print(scores)

        # Filter by confidence threshold
        valid_detections = scores > self.config.FACE_DETECTION_THRESHOLD
        valid_boxes = boxes[valid_detections]
        
        # If no faces detected, return empty array
        if len(valid_boxes) == 0:
            return np.array([])
            
        return valid_boxes
    
    def _analyze_concealment(self, frame, face_box):
        """Analyze face for signs of concealment"""
        x1, y1, x2, y2 = face_box[:4]
        face_region = frame[int(y1):int(y2), int(x1):int(x2)]
        
        if face_region.size == 0:
            return 0.0
            
        # Calculate feature visibility scores
        edge_density = cv2.Canny(face_region, 100, 200)
        edge_score = np.mean(edge_density > 0)
        
        # Calculate texture uniformity
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        texture_score = np.std(gray) / 255.0
        
        # Combine scores (lower scores indicate more concealment)
        concealment_score = 1.0 - (edge_score * 0.6 + texture_score * 0.4)
        return concealment_score
