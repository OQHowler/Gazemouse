# utils.py
import math
import numpy as np
from config import CLICK_FRAMES, DRAG_START_FRAMES, EAR_THRESHOLD, BLINK_COOLDOWN

def get_ear(landmarks, eye_indices):
    """Calculates Eye Aspect Ratio (EAR) to detect blinking."""
    # Vertical distances
    p2_p6 = math.hypot(landmarks[eye_indices[1]].x - landmarks[eye_indices[15]].x,
                       landmarks[eye_indices[1]].y - landmarks[eye_indices[15]].y)
    p3_p5 = math.hypot(landmarks[eye_indices[2]].x - landmarks[eye_indices[14]].x,
                       landmarks[eye_indices[2]].y - landmarks[eye_indices[14]].y)
    # Horizontal distance
    p1_p4 = math.hypot(landmarks[eye_indices[0]].x - landmarks[eye_indices[8]].x,
                       landmarks[eye_indices[0]].y - landmarks[eye_indices[8]].y)
    
    if p1_p4 == 0: return 0
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

class ActionTracker:
    def __init__(self):
        self.state = "IDLE"
        self.left_closed_count = 0
        self.right_closed_count = 0
        self.cooldown = 0

    def update(self, left_ear, right_ear):
        action = None
        
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        left_closed = left_ear < EAR_THRESHOLD
        right_closed = right_ear < EAR_THRESHOLD

        # 1. IGNORE BOTH EYES CLOSED (Natural Blink / Safety)
        if left_closed and right_closed:
            self.left_closed_count = 0
            self.right_closed_count = 0
            if self.state == "DRAGGING":
                self.state = "IDLE"
                return "STOP_DRAG"
            return None

        # 2. STATE: IDLE
        if self.state == "IDLE":
            # Check Left Eye
            if left_closed:
                self.left_closed_count += 1
            else:
                if CLICK_FRAMES <= self.left_closed_count < DRAG_START_FRAMES:
                    action = "LEFT_CLICK"
                    self.cooldown = BLINK_COOLDOWN
                self.left_closed_count = 0

            # Check Right Eye
            if right_closed:
                self.right_closed_count += 1
            else:
                if CLICK_FRAMES <= self.right_closed_count < DRAG_START_FRAMES:
                    action = "RIGHT_CLICK"
                    self.cooldown = BLINK_COOLDOWN
                self.right_closed_count = 0
            
            # Check Drag Start
            if self.left_closed_count >= DRAG_START_FRAMES:
                self.state = "DRAGGING"
                return "START_DRAG_LEFT"
            elif self.right_closed_count >= DRAG_START_FRAMES:
                self.state = "DRAGGING"
                return "START_DRAG_RIGHT"

        # 3. STATE: DRAGGING
        elif self.state == "DRAGGING":
            # Stop dragging if the controlling eye opens
            if self.left_closed_count > 0 and not left_closed:
                self.state = "IDLE"
                self.left_closed_count = 0
                return "STOP_DRAG"
            if self.right_closed_count > 0 and not right_closed:
                self.state = "IDLE"
                self.right_closed_count = 0
                return "STOP_DRAG"
        
        return action

class StableSmoother:
    """Simple Weighted Moving Average for cursor smoothing."""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_x = 0
        self.prev_y = 0

    def smooth(self, x, y):
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = x, y
            return x, y
        
        smooth_x = self.alpha * self.prev_x + (1 - self.alpha) * x
        smooth_y = self.alpha * self.prev_y + (1 - self.alpha) * y
        
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return smooth_x, smooth_y