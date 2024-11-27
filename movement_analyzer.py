import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque
import logging

logger = logging.getLogger(__name__)

class MovementAnalyzer:
    """Analyzes movement patterns for suspicious behavior"""
    def __init__(self, config):
        self.config = config
        self.movement_history = deque(maxlen=config.MOVEMENT_HISTORY)
        self.last_position = None
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
    def analyze_movement(self, frame, detection_box):
        """Analyze movement patterns for suspicious behavior"""
        try:
            if detection_box is None:
                return False, 0.0
                
            current_position = self._calculate_position(detection_box)
            movement_features = self._extract_movement_features(current_position)
            
            if movement_features is None:
                return False, 0.0
                
            self.movement_history.append(movement_features)
            
            if len(self.movement_history) < 10:  # Need minimum history
                return False, 0.0
                
            # Detect movement anomalies
            movement_data = np.array(list(self.movement_history))
            anomaly_scores = self.isolation_forest.fit_predict(movement_data)
            
            # Calculate anomaly probability
            anomaly_prob = np.mean(anomaly_scores == -1)
            is_anomalous = anomaly_prob > self.config.MOVEMENT_ANOMALY_THRESHOLD
            
            self.last_position = current_position
            return is_anomalous, anomaly_prob
            
        except Exception as e:
            logger.error(f"Error in movement analysis: {str(e)}")
            return False, 0.0
    
    def _calculate_position(self, detection_box):
        """Calculate center position of detection"""
        x1, y1, x2, y2 = detection_box[:4]
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _extract_movement_features(self, current_position):
        """Extract features based on movement"""
        if self.last_position is None:
            return None
            
        delta_x = current_position[0] - self.last_position[0]
        delta_y = current_position[1] - self.last_position[1]
        
        # Speed calculation (taking 1 unit as 1 meter)
        speed = np.sqrt(delta_x**2 + delta_y**2)
        
        # Return speed as feature
        return np.array([speed])
