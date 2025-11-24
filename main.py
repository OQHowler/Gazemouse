# main.py
import cv2
import mediapipe as mp
from config import SCREEN_WIDTH, SCREEN_HEIGHT, LEFT_EYE, RIGHT_EYE, SMOOTHING_FACTOR
from gaze_tracker import GazeTracker
from mouse_controller import MouseController
from utils import get_ear, ActionTracker, StableSmoother

def main():
    # 1. Setup Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    # 2. Setup MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 3. Initialize Components
    gaze_tracker = GazeTracker()
    mouse = MouseController(SCREEN_WIDTH, SCREEN_HEIGHT)
    action_tracker = ActionTracker()
    smoother = StableSmoother(alpha=SMOOTHING_FACTOR)

    print("--- Gaze Mouse Started ---")
    print("Commands:")
    print(" - Left Wink: Left Click")
    print(" - Right Wink: Right Click")
    print(" - Hold One Eye Closed: Drag Mode")
    print(" - Close Both Eyes: Reset/Safety")
    print(" - Press ESC to Quit")

    while True:
        success, frame = cap.read()
        if not success: break

        # Preprocessing
        frame = cv2.flip(frame, 1) # Mirror view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # --- 1. Action Detection ---
            left_ear = get_ear(landmarks, LEFT_EYE)
            right_ear = get_ear(landmarks, RIGHT_EYE)
            
            action = action_tracker.update(left_ear, right_ear)
            
            if action == "LEFT_CLICK":
                print("Left Click")
                mouse.left_click()
            elif action == "RIGHT_CLICK":
                print("Right Click")
                mouse.right_click()
            elif action == "START_DRAG_LEFT" or action == "START_DRAG_RIGHT":
                print("Start Drag")
                mouse.press_left()
            elif action == "STOP_DRAG":
                print("Stop Drag")
                mouse.release_left()

            # --- 2. Gaze Tracking ---
            # If dragging, use the OPEN eye for tracking. Otherwise use BOTH.
            track_mode = 'both'
            if action_tracker.state == "DRAGGING":
                if action_tracker.left_closed_count > 0: # Left is closed/holding
                    track_mode = 'right'
                else:
                    track_mode = 'left'
            
            target_x, target_y = gaze_tracker.calculate_gaze(landmarks, mode=track_mode)
            
            # --- 3. Smoothing & Move ---
            smooth_x, smooth_y = smoother.smooth(target_x, target_y)
            mouse.move(smooth_x, smooth_y)

            # --- 4. Visual Feedback (Optional) ---
            # Draw eye landmarks for debug
            for id in LEFT_EYE + RIGHT_EYE:
                lx, ly = int(landmarks[id].x * frame.shape[1]), int(landmarks[id].y * frame.shape[0])
                cv2.circle(frame, (lx, ly), 1, (0, 255, 0), -1)

        # Display
        cv2.imshow('Gaze Mouse', frame)
        if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()