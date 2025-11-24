import cv2
import mediapipe as mp
from config import SCREEN_WIDTH, SCREEN_HEIGHT, LEFT_EYE, RIGHT_EYE
from gaze_tracker import GazeTracker
from mouse_controller import MouseController
from utils import get_ear, ActionTracker, StableSmoother
# NEW IMPORT
from calibration import CalibrationManager

def main():
    cap = cv2.VideoCapture(0)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    gaze_tracker = GazeTracker()
    mouse = MouseController(SCREEN_WIDTH, SCREEN_HEIGHT)
    action_tracker = ActionTracker()
    smoother = StableSmoother()
    # NEW: Initialize Calibration
    calib_manager = CalibrationManager()

    print("--- Gaze Mouse Started ---")
    print("Press 'c' to start Calibration.")
    print("Press 'SPACE' to capture a calibration point.")
    print("Press 'ESC' to Quit.")

    while True:
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # 1. Get Raw Gaze (0.0 to 1.0)
            raw_x, raw_y = gaze_tracker.calculate_gaze(landmarks, mode='both')

            # --- CALIBRATION MODE ---
            if calib_manager.active:
                calib_manager.store_sample(raw_x, raw_y)
                frame = calib_manager.draw_ui(frame)
            
            # --- MOUSE MODE ---
            else:
                # 2. Apply Calibration Mapping (New Step!)
                # Converts raw eye math to specific screen pixels
                screen_x, screen_y = calib_manager.map_gaze(raw_x, raw_y)

                # 3. Action Detection
                left_ear = get_ear(landmarks, LEFT_EYE)
                right_ear = get_ear(landmarks, RIGHT_EYE)
                action = action_tracker.update(left_ear, right_ear)
                
                if action == "LEFT_CLICK": mouse.left_click()
                elif action == "RIGHT_CLICK": mouse.right_click()
                elif action == "START_DRAG_LEFT": mouse.press_left()
                elif action == "STOP_DRAG": mouse.release_left()

                # 4. Smoothing & Move
                # Note: We smooth the SCREEN coordinates now, not the raw ones
                smooth_x, smooth_y = smoother.smooth(screen_x, screen_y)
                
                # We normalize back to 0-1 for the mouse controller if needed, 
                # or update mouse_controller to take pixels. 
                # Let's adjust inputs to mouse.move to be normalized for safety:
                norm_x = smooth_x / SCREEN_WIDTH
                norm_y = smooth_y / SCREEN_HEIGHT
                mouse.move(norm_x, norm_y)

            # Debug circles
            for id in LEFT_EYE + RIGHT_EYE:
                lx, ly = int(landmarks[id].x * frame.shape[1]), int(landmarks[id].y * frame.shape[0])
                cv2.circle(frame, (lx, ly), 1, (0, 255, 0), -1)

        cv2.imshow('Gaze Mouse', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord('c'): # Start Calibration
            calib_manager.start_calibration()
        elif key == 32: # SPACE (Next Point)
            calib_manager.next_point()

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()