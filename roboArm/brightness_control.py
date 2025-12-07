"""
Brightness Control with Hand Gestures
Controls screen brightness using hand gestures (thumbs up/down or hand movement)
Uses MediaPipe for hand detection and screen-brightness-control for brightness adjustment
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from camera_utils import get_camera_stream

try:
    import screen_brightness_control as sbc
    BRIGHTNESS_AVAILABLE = True
except ImportError:
    BRIGHTNESS_AVAILABLE = False
    print("Warning: screen-brightness-control not available. Install with: pip install screen-brightness-control")
    print("Will display gestures but won't control brightness.")


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

def adjust_brightness(change):
    """Adjust brightness by given amount"""
    if not BRIGHTNESS_AVAILABLE:
        return
    
    try:
        current_brightness = sbc.get_brightness()[0]
        new_brightness = max(0, min(100, current_brightness + change))
        sbc.set_brightness(new_brightness)
        print(f"Brightness set to: {new_brightness}%")
    except Exception as e:
        print(f"Error adjusting brightness: {e}")

def main():
    print("Starting Brightness Control...")
    print("Instructions:")
    print("  - Thumbs Up: Increase brightness")
    print("  - Thumbs Down: Decrease brightness")
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
    
    window_name = f'{camera_type} Brightness Control'
    
    last_action_time = 0
    action_cooldown = 0.5  # 0.5 seconds between actions
    
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
        gesture_text = "No gesture"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Check for thumbs up/down
                thumb_tip = landmarks[4]
                thumb_mcp = landmarks[2]
                index_mcp = landmarks[5]
                
                # Determine if thumb is up or down
                if thumb_tip.y < thumb_mcp.y and thumb_tip.y < index_mcp.y:
                    # Thumb up
                    if current_time - last_action_time > action_cooldown:
                        adjust_brightness(5)
                        last_action_time = current_time
                    gesture_text = "Thumbs Up - Brightness +"
                    cv2.putText(frame, "BRIGHTNESS +", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                elif thumb_tip.y > thumb_mcp.y:
                    # Thumb down
                    if current_time - last_action_time > action_cooldown:
                        adjust_brightness(-5)
                        last_action_time = current_time
                    gesture_text = "Thumbs Down - Brightness -"
                    cv2.putText(frame, "BRIGHTNESS -", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display current brightness if available
        if BRIGHTNESS_AVAILABLE:
            try:
                current_brightness = sbc.get_brightness()[0]
                cv2.putText(frame, f"Brightness: {current_brightness}%", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            except:
                pass
        
        cv2.putText(frame, gesture_text, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display frame
        cv2.imshow(window_name, frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    hands.close()
    cv2.destroyAllWindows()
    print("Brightness control stopped")

if __name__ == "__main__":
    main()

