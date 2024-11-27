import logging

logger = logging.getLogger(__name__)

class ImposterDetector:
    """Combines thermal, face, and movement analysis for imposter detection"""
    def __init__(self, thermal_analyzer, face_analyzer, movement_analyzer):
        self.thermal_analyzer = thermal_analyzer
        self.face_analyzer = face_analyzer
        self.movement_analyzer = movement_analyzer
        
    def detect_imposter(self, frame, detection_box):
        """Detect potential imposter based on multiple analysis methods"""
        try:
            # Analyze each modality
            is_thermal_anomalous, thermal_score = self.thermal_analyzer.analyze_thermal_signature(frame)
            is_face_concealed, face_score = self.face_analyzer.analyze_face(frame)
            is_movement_anomalous, movement_score = self.movement_analyzer.analyze_movement(frame, detection_box)
            
            # Combine results (simple logic, can be extended)
            total_score = (thermal_score + face_score + movement_score) / 3
            is_imposter = is_thermal_anomalous or is_face_concealed or is_movement_anomalous
            
            return is_imposter, total_score
        
        except Exception as e:
            logger.error(f"Error in imposter detection: {str(e)}")
            return False, 0.0
