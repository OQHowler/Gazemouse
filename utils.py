# utils.py
import math
import time
import numpy as np
from config import CLICK_FRAMES, DRAG_START_FRAMES, EAR_THRESHOLD, BLINK_COOLDOWN, ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA

def get_ear(landmarks, eye_indices):
    """Calculates Eye Aspect Ratio (EAR) to detect blinking."""
    p2_p6 = math.hypot(landmarks[eye_indices[1]].x - landmarks[eye_indices[15]].x,
                       landmarks[eye_indices[1]].y - landmarks[eye_indices[15]].y)
    p3_p5 = math.hypot(landmarks[eye_indices[2]].x - landmarks[eye_indices[14]].x,
                       landmarks[eye_indices[2]].y - landmarks[eye_indices[14]].y)
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

        # Safety: Both closed = Reset
        if left_closed and right_closed:
            self.left_closed_count = 0
            self.right_closed_count = 0
            if self.state == "DRAGGING":
                self.state = "IDLE"
                return "STOP_DRAG"
            return None

        # State: IDLE
        if self.state == "IDLE":
            if left_closed:
                self.left_closed_count += 1
            else:
                if CLICK_FRAMES <= self.left_closed_count < DRAG_START_FRAMES:
                    action = "LEFT_CLICK"
                    self.cooldown = BLINK_COOLDOWN
                self.left_closed_count = 0

            if right_closed:
                self.right_closed_count += 1
            else:
                if CLICK_FRAMES <= self.right_closed_count < DRAG_START_FRAMES:
                    action = "RIGHT_CLICK"
                    self.cooldown = BLINK_COOLDOWN
                self.right_closed_count = 0
            
            if self.left_closed_count >= DRAG_START_FRAMES:
                self.state = "DRAGGING"
                return "START_DRAG_LEFT"
            elif self.right_closed_count >= DRAG_START_FRAMES:
                self.state = "DRAGGING"
                return "START_DRAG_RIGHT"

        # State: DRAGGING
        elif self.state == "DRAGGING":
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
    """One Euro Filter implementation for adaptive smoothing."""
    def __init__(self, min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.x_prev = None
        self.y_prev = None
        self.dx_prev = 0.0
        self.dy_prev = 0.0
        self.t_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def smooth(self, x, y):
        t_curr = time.time()
        
        if self.x_prev is None:
            self.x_prev = x
            self.y_prev = y
            self.t_prev = t_curr
            return x, y

        t_e = t_curr - self.t_prev
        
        # Avoid division by zero
        if t_e <= 0: return self.x_prev, self.y_prev

        # 1. Calculate the derivative (speed)
        dx = (x - self.x_prev) / t_e
        dy = (y - self.y_prev) / t_e

        # 2. Smooth the derivative
        a_d = self.smoothing_factor(t_e, 1.0) # 1Hz default for derivative
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        dy_hat = self.exponential_smoothing(a_d, dy, self.dy_prev)

        # 3. Calculate dynamic cutoff based on speed
        # If speed is high, cutoff is high (less smoothing)
        cutoff_x = self.min_cutoff + self.beta * abs(dx_hat)
        cutoff_y = self.min_cutoff + self.beta * abs(dy_hat)

        # 4. Smooth the signal using dynamic cutoff
        a_x = self.smoothing_factor(t_e, cutoff_x)
        a_y = self.smoothing_factor(t_e, cutoff_y)

        x_hat = self.exponential_smoothing(a_x, x, self.x_prev)
        y_hat = self.exponential_smoothing(a_y, y, self.y_prev)

        # Update previous values
        self.x_prev = x_hat
        self.y_prev = y_hat
        self.dx_prev = dx_hat
        self.dy_prev = dy_hat
        self.t_prev = t_curr

        return x_hat, y_hat