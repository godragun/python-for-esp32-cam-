"""
Open Chrome with 5 Fingers Gesture
Detects when 5 fingers are shown and opens Chrome browser
Uses MediaPipe for hand detection
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import subprocess
import os
from camera_utils import get_camera_stream

def count_fingers(landmarks):
    """Count extended fingers"""
    tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    fingers = []
    
    # Thumb (different logic - compare x coordinates)
    if landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Other fingers
    for id in range(1, 5):
        if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers.count(1)

def open_chrome():
    """Open Chrome browser"""
    try:
        # Try different methods to open Chrome based on OS
        if os.name == 'nt':  # Windows
            chrome_paths = [
                r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
                r'C:\Users\{}\AppData\Local\Google\Chrome\Application\chrome.exe'.format(os.getenv('USERNAME'))
            ]
            for path in chrome_paths:
                if os.path.exists(path):
                    subprocess.Popen([path])
                    return True
            # Fallback: try using start command
            subprocess.Popen(['start', 'chrome'], shell=True)
            return True
        else:  # Linux/Mac
            subprocess.Popen(['google-chrome'])
            return True
    except Exception as e:
        print(f"Error opening Chrome: {e}")
        return False

def main():
    print("Starting Chrome Opener (5 Fingers Gesture)...")
    print("Instructions: Show 5 fingers to open Chrome browser")
    print("Press 'q' to quit")
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    stream_gen, camera_type = get_camera_stream()
    if stream_gen is None:
        return
    
    window_name = f'{camera_type} Chrome Opener'
    
    last_action_time = 0
    action_cooldown = 2.0  # 2 seconds cooldown to prevent multiple opens
    chrome_opened = False
    
    for frame in stream_gen:
        if frame is None:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = hands.process(rgb_frame)
        
        current_time = time.time()
        finger_count = 0
        status_text = "Show 5 fingers to open Chrome"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Count fingers
                finger_count = count_fingers(landmarks)
                
                # Check if 5 fingers are shown
                if finger_count == 5:
                    if current_time - last_action_time > action_cooldown:
                        print("5 fingers detected! Opening Chrome...")
                        if open_chrome():
                            chrome_opened = True
                            status_text = "Chrome opened!"
                            print("Chrome opened successfully!")
                        else:
                            status_text = "Failed to open Chrome"
                        last_action_time = current_time
        
        # Display finger count
        cv2.putText(frame, f"Fingers: {finger_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display status
        if finger_count == 5:
            cv2.putText(frame, "5 FINGERS DETECTED!", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Need: {5 - finger_count} more fingers", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.putText(frame, status_text, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if chrome_opened:
            cv2.putText(frame, "Chrome Status: OPENED", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow(window_name, frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    hands.close()
    cv2.destroyAllWindows()
    print("Chrome opener stopped")

if __name__ == "__main__":
    main()

