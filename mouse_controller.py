# mouse_controller.py
import pyautogui

class MouseController:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        pyautogui.FAILSAFE = False # Warning: Move mouse to corner won't kill script
        pyautogui.PAUSE = 0

    def move(self, x, y):
        """Moves cursor to normalized coordinates (0.0-1.0)."""
        screen_x = int(self.screen_width * x)
        screen_y = int(self.screen_height * y)
        
        # Ensure bounds
        screen_x = max(0, min(screen_x, self.screen_width - 1))
        screen_y = max(0, min(screen_y, self.screen_height - 1))
        
        pyautogui.moveTo(screen_x, screen_y, duration=0)

    def left_click(self):
        pyautogui.click(button='left')

    def right_click(self):
        pyautogui.click(button='right')

    def press_left(self):
        pyautogui.mouseDown(button='left')

    def release_left(self):
        pyautogui.mouseUp(button='left')