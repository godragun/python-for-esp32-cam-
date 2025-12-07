"""
Drawing Application with Color Selection
Draw on screen using hand gestures with 6 selectable colors
Uses MediaPipe for hand tracking and finger gestures to select colors
"""

import cv2
import numpy as np
import mediapipe as mp
from camera_utils import get_camera_stream

# ESP32-CAM Configuration
WIFI_NAME = "harendra cam free wifi"

# Define 6 colors (BGR format for OpenCV)
COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 0, 255),    # Magenta
    (255, 255, 0),    # Cyan
    (0, 165, 255),    # Orange
]

COLOR_NAMES = ["Blue", "Green", "Red", "Magenta", "Cyan", "Orange"]

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

def get_color_selector_region(frame, h, w):
    """Create color selector region at the top"""
    selector_height = 80
    selector_y = 0
    
    # Draw color palette
    color_width = w // 6
    for i, color in enumerate(COLORS):
        x_start = i * color_width
        x_end = (i + 1) * color_width
        cv2.rectangle(frame, (x_start, selector_y), (x_end, selector_height), color, -1)
        cv2.rectangle(frame, (x_start, selector_y), (x_end, selector_height), (255, 255, 255), 2)
        
        # Draw color name
        text_size = cv2.getTextSize(COLOR_NAMES[i], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = x_start + (color_width - text_size[0]) // 2
        text_y = selector_height // 2 + 5
        cv2.putText(frame, COLOR_NAMES[i], (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return selector_y, selector_height

def is_in_color_selector(x, y, selector_y, selector_height, w):
    """Check if point is in color selector region"""
    return selector_y <= y <= selector_y + selector_height and 0 <= x <= w

def get_selected_color_index(x, w):
    """Get color index based on x position"""
    color_width = w // 6
    return min(5, max(0, x // color_width))

def main():
    print("Starting Drawing Application...")
    print("Instructions:")
    print("  - Use index finger to draw")
    print("  - Show 1-6 fingers to select colors (1=Blue, 2=Green, 3=Red, 4=Magenta, 5=Cyan, 6=Orange)")
    print("  - Show fist (0 fingers) to clear canvas")
    print("  - Point at top color bar to select color by position")
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
    
    window_name = f'{camera_type} Drawing App'
    
    # Drawing variables
    current_color_index = 0
    drawing = False
    canvas = None
    prev_x, prev_y = None, None
    
    for frame in stream_gen:
        if frame is None:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Initialize canvas
        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Create color selector region
        selector_y, selector_height = get_color_selector_region(frame, h, w)
        drawing_area_y = selector_height
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = hands.process(rgb_frame)
        
        index_tip_x, index_tip_y = None, None
        finger_count = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                
                # Get index finger tip position
                index_tip = landmarks[8]
                index_tip_x = int(index_tip.x * w)
                index_tip_y = int(index_tip.y * h)
                
                # Count fingers
                finger_count = count_fingers(landmarks)
                
                # Color selection by finger count (1-6)
                if 1 <= finger_count <= 6:
                    current_color_index = finger_count - 1
                
                # Check if pointing at color selector
                if is_in_color_selector(index_tip_x, index_tip_y, selector_y, selector_height, w):
                    current_color_index = get_selected_color_index(index_tip_x, w)
                
                # Clear canvas with fist (0 fingers)
                if finger_count == 0:
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Drawing logic (only when index finger is extended)
                if index_tip_x and index_tip_y:
                    # Only draw if finger is in drawing area (below selector)
                    if index_tip_y > drawing_area_y:
                        # Check if index finger is up (for drawing)
                        index_tip_landmark = landmarks[8]
                        index_pip_landmark = landmarks[6]
                        
                        # Index finger is up if tip is above PIP
                        if index_tip_landmark.y < index_pip_landmark.y:
                            if prev_x is not None and prev_y is not None:
                                # Draw line from previous point to current point
                                cv2.line(canvas, (prev_x, prev_y), 
                                        (index_tip_x, index_tip_y), 
                                        COLORS[current_color_index], 5)
                                cv2.line(frame, (prev_x, prev_y), 
                                        (index_tip_x, index_tip_y), 
                                        COLORS[current_color_index], 5)
                            prev_x, prev_y = index_tip_x, index_tip_y
                            drawing = True
                        else:
                            prev_x, prev_y = None, None
                            drawing = False
                    else:
                        prev_x, prev_y = None, None
                else:
                    prev_x, prev_y = None, None
                    drawing = False
                
                # Draw hand landmarks (optional, can be commented out for cleaner view)
                # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            prev_x, prev_y = None, None
            drawing = False
        
        # Merge canvas with current frame
        # Only show canvas in drawing area
        canvas_roi = canvas[drawing_area_y:, :]
        frame_roi = frame[drawing_area_y:, :]
        combined_roi = cv2.addWeighted(frame_roi, 0.7, canvas_roi, 0.3, 0)
        frame[drawing_area_y:, :] = combined_roi
        
        # Highlight selected color in selector
        color_width = w // 6
        selected_x_start = current_color_index * color_width
        selected_x_end = (current_color_index + 1) * color_width
        cv2.rectangle(frame, (selected_x_start, selector_y), 
                     (selected_x_end, selector_height), (255, 255, 255), 3)
        
        # Display current color and finger count
        cv2.putText(frame, f"Color: {COLOR_NAMES[current_color_index]}", (10, h - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[current_color_index], 2)
        cv2.putText(frame, f"Fingers: {finger_count}", (10, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show drawing indicator
        if drawing:
            cv2.putText(frame, "DRAWING", (w - 150, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow(window_name, frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    hands.close()
    cv2.destroyAllWindows()
    print("Drawing application stopped")

if __name__ == "__main__":
    main()

