"""
Volume Control with Hand Gestures
Controls system volume using hand gestures (thumbs up/down or vertical hand movement)
Uses MediaPipe for hand detection and pycaw for volume control
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from camera_utils import get_camera_stream

try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    import comtypes
    VOLUME_AVAILABLE = True
    
    # Initialize volume control
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
except ImportError:
    VOLUME_AVAILABLE = False
    print("Warning: pycaw not available. Install with: pip install pycaw")
    print("Will display gestures but won't control volume.")
except Exception as e:
    VOLUME_AVAILABLE = False
    print(f"Warning: Volume control initialization failed: {e}")


def adjust_volume(change_percent):
    """Adjust volume by given percentage"""
    if not VOLUME_AVAILABLE:
        return
    
    try:
        current_volume = volume.GetMasterVolumeLevelScalar()
        new_volume = max(0.0, min(1.0, current_volume + change_percent))
        volume.SetMasterVolumeLevelScalar(new_volume, None)
        print(f"Volume set to: {int(new_volume * 100)}%")
    except Exception as e:
        print(f"Error adjusting volume: {e}")

def get_volume_level():
    """Get current volume level"""
    if not VOLUME_AVAILABLE:
        return 0
    try:
        return int(volume.GetMasterVolumeLevelScalar() * 100)
    except:
        return 0

def main():
    print("Starting Volume Control...")
    print("Instructions:")
    print("  - Thumbs Up: Increase volume")
    print("  - Thumbs Down: Decrease volume")
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
    
    window_name = f'{camera_type} Volume Control'
    
    last_action_time = 0
    action_cooldown = 0.3  # 0.3 seconds between actions
    
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
                        adjust_volume(0.05)  # Increase by 5%
                        last_action_time = current_time
                    gesture_text = "Thumbs Up - Volume +"
                    cv2.putText(frame, "VOLUME +", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                elif thumb_tip.y > thumb_mcp.y:
                    # Thumb down
                    if current_time - last_action_time > action_cooldown:
                        adjust_volume(-0.05)  # Decrease by 5%
                        last_action_time = current_time
                    gesture_text = "Thumbs Down - Volume -"
                    cv2.putText(frame, "VOLUME -", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display current volume
        current_volume = get_volume_level()
        cv2.putText(frame, f"Volume: {current_volume}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(frame, gesture_text, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display frame
        cv2.imshow(window_name, frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    hands.close()
    cv2.destroyAllWindows()
    print("Volume control stopped")

if __name__ == "__main__":
    main()

