"""
Face Emotion Recognition
Connects to ESP32-CAM and recognizes emotions from faces
Uses FER (Facial Expression Recognition) library or custom model
"""

import cv2
import numpy as np
from fer import FER
from camera_utils import get_camera_stream

def main():
    print("Starting Face Emotion Recognition...")
    print("Press 'q' to quit")
    
    # Initialize emotion detector
    try:
        print("Loading FER model (this may take a moment on first run)...")
        emotion_detector = FER(mtcnn=True)
        print("FER model loaded successfully")
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR: FER model could not be loaded!")
        print(f"{'='*60}")
        print(f"Error details: {e}")
        print("\nTo fix this, please install the required dependencies:")
        print("  python -m pip install fer")
        print("\nNote: This will also install PyTorch (large download ~200MB)")
        print("      Please be patient during installation.")
        print(f"{'='*60}\n")
        return
    
    stream_gen, camera_type = get_camera_stream()
    if stream_gen is None:
        return
    
    window_name = f'{camera_type} Emotion Recognition'
    
    frame_count = 0
    for frame in stream_gen:
        if frame is None:
            continue
        
        # Process every 5th frame for better performance
        if frame_count % 5 == 0:
            try:
                # Detect emotions
                emotions = emotion_detector.detect_emotions(frame)
                
                # Draw bounding boxes and emotions
                if emotions:
                    for face_data in emotions:
                        bounding_box = face_data["box"]
                        emotions_dict = face_data["emotions"]
                        
                        # Get top emotion
                        top_emotion = max(emotions_dict, key=emotions_dict.get)
                        emotion_score = emotions_dict[top_emotion]
                        
                        x, y, w, h = bounding_box
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Display emotion and score
                        text = f"{top_emotion}: {emotion_score:.2f}"
                        cv2.putText(frame, text, (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Display all emotions
                        y_offset = y + h + 20
                        for emotion, score in sorted(emotions_dict.items(), 
                                                    key=lambda x: x[1], reverse=True)[:3]:
                            emotion_text = f"{emotion}: {score:.2f}"
                            cv2.putText(frame, emotion_text, (x, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            y_offset += 15
            except Exception as e:
                print(f"Emotion detection error: {e}")
        
        frame_count += 1
        
        # Display frame
        cv2.imshow(window_name, frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("Emotion recognition stopped")

if __name__ == "__main__":
    main()

