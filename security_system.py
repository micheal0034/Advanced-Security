# import cv2
# import logging
# from config import SecurityConfig
# from thermal_analyzer import ThermalAnalyzer
# from face_analyzer import FaceAnalyzer
# from movement_analyzer import MovementAnalyzer
# from imposter_detector import ImposterDetector
# from ultralytics import YOLO

# logger = logging.getLogger(__name__)

# class SecuritySystem:
#     """Main security system for imposter detection"""
#     def __init__(self):
#         self.config = SecurityConfig()
#         self.thermal_analyzer = ThermalAnalyzer(self.config)
#         self.face_analyzer = FaceAnalyzer(self.config)
#         self.movement_analyzer = MovementAnalyzer(self.config)
#         self.imposter_detector = ImposterDetector(self.thermal_analyzer, self.face_analyzer, self.movement_analyzer)
        
#     def start_monitoring(self):
#         """Start video monitoring and detection"""
#         try:
#             # Initialize video capture
#             cap = cv2.VideoCapture(camera_url)
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 # Use YOLOv8 to detect people in frame
#                 self.model = YOLO('yolov8n.pt')
                
#                 results = self.model(frame, model='yolov8n')
                
#                 # Get bounding box coordinates of detected person
#                 detection_box = None
#                 for r in results:
#                     boxes = r.boxes
#                     for box in boxes:
#                         # YOLOv8 returns class predictions - check if person (class 0)
#                         if box.cls == 0:
#                             # Convert box coordinates to [x1,y1,x2,y2] format
#                             x1, y1, x2, y2 = box.xyxy[0]
#                             detection_box = [int(x1), int(y1), int(x2), int(y2)]
#                             print(detection_box)
#                             break
#                     if detection_box:
#                         break
#                 # detection_box = [100, 100, 330, 330]
                
#                 # Skip detection if no person found
#                 if not detection_box:
#                     continue
                
#                 is_imposter, score = self.imposter_detector.detect_imposter(frame, detection_box)
#                 if is_imposter:
#                     logger.warning(f"Imposter detected! Score: {score}")
                
#                 # Display frame for monitoring
#                 cv2.imshow("Security System", frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
            
#             cap.release()
#             cv2.destroyAllWindows()
        
#         except Exception as e:
#             logger.error(f"Error in security system: {str(e)}")



import cv2
import logging
from config import SecurityConfig
from thermal_analyzer import ThermalAnalyzer
from face_analyzer import FaceAnalyzer
from movement_analyzer import MovementAnalyzer
from imposter_detector import ImposterDetector
from ultralytics import YOLO

logger = logging.getLogger(__name__)

print('......... Loading Security System')
class SecuritySystem:
    """Main security system for imposter detection"""
    def __init__(self):
        self.config = SecurityConfig()
        self.thermal_analyzer = ThermalAnalyzer(self.config)
        self.face_analyzer = FaceAnalyzer(self.config)
        self.movement_analyzer = MovementAnalyzer(self.config)
        self.imposter_detector = ImposterDetector(self.thermal_analyzer, self.face_analyzer, self.movement_analyzer)
        self.model = YOLO('yolov8n.pt')  # Load YOLO model once during initialization
        
    def start_monitoring(self, camera_url="0"):
        """Start video monitoring and detection
        
        Args:
            camera_url (str): URL of the network camera or "0" for local webcam.
        """
        print('......... Loading Camera System')
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(camera_url)
            if not cap.isOpened():
                logger.error("Unable to connect to the camera. Please check the URL.")
                return
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Use YOLOv8 to detect people in frame
                results = self.model(frame, model='yolov8n')
                
                # Get bounding box coordinates of detected person
                detection_box = None
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # YOLOv8 returns class predictions - check if person (class 0)
                        if box.cls == 0:
                            # Convert box coordinates to [x1, y1, x2, y2] format
                            x1, y1, x2, y2 = box.xyxy[0]
                            detection_box = [int(x1), int(y1), int(x2), int(y2)]
                            logger.info(f"Detection box: {detection_box}")
                            break
                    if detection_box:
                        break
                
                # Skip detection if no person found
                if not detection_box:
                    continue
                
                # Analyze the detected person
                is_imposter, score = self.imposter_detector.detect_imposter(frame, detection_box)
                if is_imposter:
                    logger.warning(f"Imposter detected! Score: {score}")
                
                # Display frame for monitoring
                cv2.imshow("Security System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        except Exception as e:
            logger.error(f"Error in security system: {str(e)}")
