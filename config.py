# config.py

import pyautogui

# Screen Dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Camera Settings
CAM_WIDTH, CAM_HEIGHT = 640, 480

# Eye Landmark Indices (MediaPipe)
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Thresholds
EAR_THRESHOLD = 0.22        # Eye Aspect Ratio to consider eye closed
CLICK_FRAMES = 2            # Frames to register a click (fast blink)
DRAG_START_FRAMES = 12      # Frames to register a drag (long hold)
BLINK_COOLDOWN = 5          # Frames to wait before registering another action

# Smoothing (0.0 = no smoothing, 0.9 = very slow/smooth)
SMOOTHING_FACTOR = 0.6 

# Calibration / Sensitivity
# Adjust these to map the eye movement range to the screen
# Higher = more sensitive, Lower = requires more eye movement
SENSITIVITY = 2.5