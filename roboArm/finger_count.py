"""
Finger Counting Detector
Counts the number of extended fingers and displays the count
Uses MediaPipe for hand detection
"""

import cv2
import numpy as np
import mediapipe as mp
from camera_utils import get_camera_stream

def count_fingers(landmarks):
    """Count extended fingers"""
    tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    fingers = []
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    
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
    
    total_count = fingers.count(1)
    extended_fingers = [finger_names[i] for i, val in enumerate(fingers) if val == 1]
    
    return total_count, extended_fingers

def main():
    print("Starting Finger Counter...")
    print("Instructions: Show your hand to count fingers")
    print("Press 'q' to quit")
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Can detect both hands
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    stream_gen, camera_type = get_camera_stream()
    if stream_gen is None:
        return
    
    window_name = f'{camera_type} Finger Counter'
    
    for frame in stream_gen:
        if frame is None:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = hands.process(rgb_frame)
        
        total_fingers = 0
        all_extended = []
        
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks = hand_landmarks.landmark
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Count fingers for this hand
                finger_count, extended_fingers = count_fingers(landmarks)
                total_fingers += finger_count
                all_extended.extend(extended_fingers)
                
                # Display count for each hand
                hand_label = f"Hand {idx + 1}: {finger_count} fingers"
                cv2.putText(frame, hand_label, (10, 60 + idx * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display total finger count
        cv2.putText(frame, f"Total Fingers: {total_fingers}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Display extended fingers if any
        if all_extended:
            fingers_text = ", ".join(all_extended)
            cv2.putText(frame, f"Extended: {fingers_text}", (10, h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display hand count
        if results.multi_hand_landmarks:
            cv2.putText(frame, f"Hands Detected: {len(results.multi_hand_landmarks)}", 
                       (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No Hands Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow(window_name, frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    hands.close()
    cv2.destroyAllWindows()
    print("Finger counter stopped")

if __name__ == "__main__":
    main()

