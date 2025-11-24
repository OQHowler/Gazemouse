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

# Blink & Action Thresholds
EAR_THRESHOLD = 0.22        # Eye Aspect Ratio to consider eye closed
CLICK_FRAMES = 2            # Frames to register a click
DRAG_START_FRAMES = 12      # Frames to register a drag
BLINK_COOLDOWN = 5          # Frames to wait between actions

# Calibration / Sensitivity
# Adjust these to map the eye movement range to the screen
# Higher = more sensitive, Lower = requires more eye movement
SENSITIVITY = 2.0  

# --- ONE EURO FILTER SETTINGS (Jitter Reduction) ---
# MIN_CUTOFF: Lower = more smoothing (steadier cursor), but more lag.
# BETA: Higher = less lag when moving fast.
ONE_EURO_MIN_CUTOFF = 0.01  
ONE_EURO_BETA = 1.5