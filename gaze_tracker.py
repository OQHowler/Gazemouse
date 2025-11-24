# gaze_tracker.py
import numpy as np
from config import LEFT_EYE, RIGHT_EYE, LEFT_IRIS, RIGHT_IRIS, SENSITIVITY

class GazeTracker:
    def calculate_gaze(self, landmarks, mode='both'):
        """
        Returns normalized (x, y) coordinates [0.0, 1.0] based on iris position.
        """
        left_pos = self._get_iris_pos(landmarks, LEFT_EYE, LEFT_IRIS)
        right_pos = self._get_iris_pos(landmarks, RIGHT_EYE, RIGHT_IRIS)

        if mode == 'left':
            avg_pos = left_pos
        elif mode == 'right':
            avg_pos = right_pos
        else: # 'both'
            avg_pos = ((left_pos[0] + right_pos[0]) / 2, (left_pos[1] + right_pos[1]) / 2)

        # Apply sensitivity and centering
        # We assume center of eye is 0.5, 0.5. We subtract 0.5, scale, and add 0.5 back.
        gaze_x = 0.5 + (avg_pos[0] - 0.5) * SENSITIVITY
        gaze_y = 0.5 + (avg_pos[1] - 0.5) * SENSITIVITY

        # Clamp values to screen edges
        return max(0.0, min(1.0, gaze_x)), max(0.0, min(1.0, gaze_y))

    def _get_iris_pos(self, landmarks, eye_indices, iris_indices):
        """Calculates iris position relative to eye width/height."""
        
        # Eye corners (Inner and Outer)
        # For Left Eye: 362 (inner), 263 (outer) - Simplified approximation
        # Using 0 and 8 index from our config list corresponding to corners
        p_left = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
        p_right = np.array([landmarks[eye_indices[8]].x, landmarks[eye_indices[8]].y])
        
        # Iris Center
        iris_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in iris_indices], axis=0)

        # Eye Width and Height Vectors
        eye_width = np.linalg.norm(p_right - p_left)
        
        # Center of the eye span
        eye_center = (p_left + p_right) / 2.0

        # Calculate relative position (0.5 is center)
        # Note: This is a simplified projection for 2D screens
        rel_x = 0.5 + (iris_center[0] - eye_center[0]) / eye_width
        rel_y = 0.5 + (iris_center[1] - eye_center[1]) / (eye_width * 0.3) # Height is approx 30% of width

        return rel_x, rel_y