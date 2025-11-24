# calibration.py
import numpy as np
import cv2
from config import SCREEN_WIDTH, SCREEN_HEIGHT

class CalibrationManager:
    def __init__(self):
        # Define 5 points: Top-Left, Top-Right, Center, Bottom-Left, Bottom-Right
        # We perform a margin inset so you aren't looking at the absolute edge
        m = 50 # margin
        self.points = [
            (m, m),                         # Top-Left
            (SCREEN_WIDTH - m, m),          # Top-Right
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), # Center
            (m, SCREEN_HEIGHT - m),         # Bottom-Left
            (SCREEN_WIDTH - m, SCREEN_HEIGHT - m)    # Bottom-Right
        ]
        self.current_point_idx = 0
        self.eye_samples = []  # Stores eye (x,y) for the current point
        self.calibration_map = [] # Stores (eye_x, eye_y) -> (screen_x, screen_y)
        self.matrix = None # The Magic Math Matrix
        self.active = False

    def start_calibration(self):
        self.active = True
        self.current_point_idx = 0
        self.eye_samples = []
        self.calibration_map = []
        self.matrix = None
        print("Calibration Started! Look at the red dot and press SPACE.")

    def store_sample(self, eye_x, eye_y):
        """Called every frame while user looks at a dot."""
        self.eye_samples.append((eye_x, eye_y))

    def next_point(self):
        """Called when user presses SPACE to confirm they looked at the dot."""
        if not self.active: return

        # Average the eye samples for better accuracy (removes wobble!)
        if not self.eye_samples:
            print("No data collected for this point! Look at the screen.")
            return

        avg_eye = np.mean(self.eye_samples, axis=0)
        target_screen = self.points[self.current_point_idx]
        
        self.calibration_map.append((avg_eye, target_screen))
        print(f"Point {self.current_point_idx+1} captured: Eye{avg_eye} -> Screen{target_screen}")

        self.eye_samples = [] # Reset for next point
        self.current_point_idx += 1

        # Check if we are done
        if self.current_point_idx >= len(self.points):
            self.compute_matrix()
            self.active = False
            print("Calibration Complete!")

    def compute_matrix(self):
        """Calculates the Homography Matrix using the collected points."""
        if len(self.calibration_map) < 4:
            print("Not enough points for calibration.")
            return

        # Prepare data for cv2.findHomography
        src_points = np.array([p[0] for p in self.calibration_map], dtype=np.float32)
        dst_points = np.array([p[1] for p in self.calibration_map], dtype=np.float32)

        # Calculate the matrix
        # RANSAC helps ignore outliers (blinks/glitches during calibration)
        self.matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)

    def map_gaze(self, eye_x, eye_y):
        """Applies the matrix to convert Eye X,Y to Screen X,Y."""
        if self.matrix is None:
            # Fallback to simple scaling if not calibrated
            return eye_x * SCREEN_WIDTH, eye_y * SCREEN_HEIGHT
        
        # Apply Homography
        # We need a generic [x, y, 1] vector for matrix multiplication
        vector = np.array([[[eye_x, eye_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(vector, self.matrix)
        
        return transformed[0][0][0], transformed[0][0][1]

    def draw_ui(self, frame):
        """Draws the calibration dot on the camera feed."""
        if not self.active: return frame
        
        # We can't draw on the real screen easily with OpenCV, 
        # so we give visual cues in the camera window.
        
        # Get the normalized position of the current target point
        target = self.points[self.current_point_idx]
        norm_x = int((target[0] / SCREEN_WIDTH) * frame.shape[1])
        norm_y = int((target[1] / SCREEN_HEIGHT) * frame.shape[0])

        # Draw a big red circle
        cv2.circle(frame, (norm_x, norm_y), 10, (0, 0, 255), -1)
        cv2.putText(frame, "LOOK HERE & PRESS SPACE", (norm_x - 100, norm_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame