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
        # We subtract 0.5 to center it, scale it, and add 0.5 back
        gaze_x = 0.5 + (avg_pos[0] - 0.5) * SENSITIVITY
        gaze_y = 0.5 + (avg_pos[1] - 0.5) * SENSITIVITY

        # Clamp values to ensure it stays on screen
        return max(0.0, min(1.0, gaze_x)), max(0.0, min(1.0, gaze_y))

    def _get_iris_pos(self, landmarks, eye_indices, iris_indices):
        """Calculates iris position relative to rigid eye landmarks."""
        
        # 1. Get the Iris Center (The actual eyeball)
        iris_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in iris_indices], axis=0)

        # 2. Get Rigid Reference Points (Corners)
        # 0 = inner corner, 8 = outer corner in our config lists
        p_inner = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
        p_outer = np.array([landmarks[eye_indices[8]].x, landmarks[eye_indices[8]].y])
        
        # 3. Get Eyelid Reference Points (Top and Bottom)
        # 12 = center of top lid, 4 = center of bottom lid in our config lists
        p_top = np.array([landmarks[eye_indices[12]].x, landmarks[eye_indices[12]].y])
        p_bottom = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])

        # --- HORIZONTAL CALCULATION (X) ---
        # Distance from inner corner to outer corner
        eye_width = np.linalg.norm(p_outer - p_inner)
        if eye_width == 0: return 0.5, 0.5
        
        # Project iris center onto the line connecting corners
        # Simple projection: how far is the iris along the width?
        # (This vector math projects the iris position onto the horizontal axis)
        eye_vec = p_outer - p_inner
        iris_vec = iris_center - p_inner
        # Dot product / magnitude gives projection
        proj_x = np.dot(iris_vec, eye_vec) / (eye_width * eye_width)

        # --- VERTICAL CALCULATION (Y) ---
        # Instead of generic height, use distance from Top Eyelid vs Bottom Eyelid
        dist_top = np.linalg.norm(iris_center - p_top)
        dist_bottom = np.linalg.norm(iris_center - p_bottom)
        total_height = dist_top + dist_bottom

        if total_height == 0: return 0.5, 0.5
        
        # Normalized Y (0.0 = top, 1.0 = bottom)
        rel_y = dist_top / total_height

        return proj_x, rel_y