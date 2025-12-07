"""
Face Recognition with Color-Coded Bounding Boxes
Connects to ESP32-CAM and detects faces with color coding based on number of heads detected
Colors: Purple (1), Green (2), Yellow (3), Red (4+)
"""

import cv2
import numpy as np
from camera_utils import get_camera_stream

def get_color_for_count(count):
    """Return color based on number of faces detected"""
    colors = {
        1: (255, 0, 255),  # Purple (BGR)
        2: (0, 255, 0),    # Green
        3: (0, 255, 255),  # Yellow
    }
    return colors.get(count, (0, 0, 255)) if count <= 3 else (0, 0, 255)  # Red for 4+

def main():
    print("Starting Face Recognition...")
    print("Press 'q' to quit")
    
    # Load face detection model (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        return
    
    stream_gen, camera_type = get_camera_stream()
    if stream_gen is None:
        return
    
    window_name = f'{camera_type} Face Recognition'
    
    frame_count = 0
    for frame in stream_gen:
        if frame is None:
            continue
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Get color based on number of faces
        face_count = len(faces)
        color = get_color_for_count(face_count)
        
        # Draw bounding boxes around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"Face {face_count}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display face count on frame
        cv2.putText(frame, f"Faces Detected: {face_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Display frame
        cv2.imshow(window_name, frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("Face recognition stopped")

if __name__ == "__main__":
    main()

