"""
Red Dot Capture Game
A game where red dots appear randomly and you capture them using hand gestures
Uses hand detection to track hand position and detect taps/clicks on dots
"""

import cv2
import numpy as np
import mediapipe as mp
import random
import time
from camera_utils import get_camera_stream

class RedDot:
    def __init__(self, x, y, radius=15):
        self.x = x
        self.y = y
        self.radius = radius
        self.active = True
        self.spawn_time = time.time()
    
    def draw(self, frame):
        if self.active:
            cv2.circle(frame, (self.x, self.y), self.radius, (0, 0, 255), -1)
            cv2.circle(frame, (self.x, self.y), self.radius, (255, 255, 255), 2)
    
    def is_collision(self, hand_x, hand_y, threshold=30):
        distance = np.sqrt((self.x - hand_x)**2 + (self.y - hand_y)**2)
        return distance < (self.radius + threshold)


def main():
    print("Starting Red Dot Capture Game...")
    print("Instructions: Use your index finger to tap red dots!")
    print("Press 'q' to quit")
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    stream_gen, camera_type = get_camera_stream()
    if stream_gen is None:
        return
    
    window_name = f'{camera_type} Red Dot Game'
    
    # Game variables
    dots = []
    score = 0
    last_dot_time = time.time()
    dot_spawn_interval = 2.0  # Spawn new dot every 2 seconds
    frame_width = 640
    frame_height = 480
    
    for frame in stream_gen:
        if frame is None:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_width, frame_height = w, h
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = hands.process(rgb_frame)
        
        # Get hand position (index finger tip)
        hand_x, hand_y = None, None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get index finger tip (landmark 8)
                index_tip = hand_landmarks.landmark[8]
                hand_x = int(index_tip.x * w)
                hand_y = int(index_tip.y * h)
                
                # Draw index finger tip
                cv2.circle(frame, (hand_x, hand_y), 10, (0, 255, 0), -1)
        
        # Spawn new dots
        current_time = time.time()
        if current_time - last_dot_time > dot_spawn_interval:
            new_x = random.randint(50, w - 50)
            new_y = random.randint(50, h - 50)
            dots.append(RedDot(new_x, new_y))
            last_dot_time = current_time
        
        # Update and draw dots
        dots_to_remove = []
        for i, dot in enumerate(dots):
            # Remove dots older than 5 seconds
            if time.time() - dot.spawn_time > 5:
                dots_to_remove.append(i)
                continue
            
            # Check collision with hand
            if hand_x and hand_y and dot.is_collision(hand_x, hand_y):
                dots_to_remove.append(i)
                score += 10
                print(f"Dot captured! Score: {score}")
            
            dot.draw(frame)
        
        # Remove captured/expired dots
        for i in reversed(dots_to_remove):
            dots.pop(i)
        
        # Draw score
        cv2.putText(frame, f"Score: {score}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Dots: {len(dots)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw instruction
        cv2.putText(frame, "Tap red dots with your finger!", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display frame
        cv2.imshow(window_name, frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    hands.close()
    cv2.destroyAllWindows()
    print(f"Game over! Final score: {score}")

if __name__ == "__main__":
    main()

