import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque
import logging

logger = logging.getLogger(__name__)

class ThermalAnalyzer:
    """Analyzes thermal patterns to detect anomalies"""
    def __init__(self, config):
        self.config = config
        self.thermal_history = deque(maxlen=config.THERMAL_WINDOW_SIZE)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
    def analyze_thermal_signature(self, frame):
        """Analyze thermal signature for anomalies"""
        try:
            # Extract thermal features
            thermal_features = self._extract_thermal_features(frame)
            self.thermal_history.append(thermal_features)
            
            if len(self.thermal_history) < self.config.THERMAL_WINDOW_SIZE:
                return False, 0.0
            
            # Detect thermal anomalies
            thermal_data = np.array(list(self.thermal_history))
            anomaly_scores = self.isolation_forest.fit_predict(thermal_data)
            print(anomaly_scores)
            print(thermal_data)
            
            # Calculate anomaly probability
            anomaly_prob = np.mean(anomaly_scores == -1)  # -1 indicates anomaly
            is_anomalous = anomaly_prob > self.config.THERMAL_ANOMALY_THRESHOLD
            
            return is_anomalous, anomaly_prob
            
        except Exception as e:
            logger.error(f"Error in thermal analysis: {str(e)}")
            return False, 0.0
    
    def _extract_thermal_features(self, frame):
        """Extract thermal features from frame"""
        temp_frame = frame.astype(float) * 0.1
        
        features = []
        features.extend([np.mean(temp_frame), np.std(temp_frame), np.max(temp_frame), np.min(temp_frame), np.median(temp_frame)])
        
        # Calculate temperature gradient
        gradient_x = np.gradient(temp_frame, axis=1)
        gradient_y = np.gradient(temp_frame, axis=0)
        features.extend([np.mean(np.abs(gradient_x)), np.mean(np.abs(gradient_y))])
        
        return np.array(features)
