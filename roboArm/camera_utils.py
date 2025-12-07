"""
Camera Utility Module
Provides functions to connect to ESP32-CAM with fallback to laptop camera
"""

import cv2
import numpy as np
import urllib.request
import urllib.error

# ESP32-CAM Configuration
ESP32_CAM_IP = "192.168.4.1"
STREAM_URL = f"http://{ESP32_CAM_IP}/stream"
WIFI_NAME = "harendra cam free wifi"

def get_esp32_stream():
    """Get ESP32-CAM stream generator"""
    try:
        stream = urllib.request.urlopen(STREAM_URL, timeout=5)
        bytes_data = bytes()
        while True:
            bytes_data += stream.read(1024)
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame
    except (urllib.error.URLError, Exception) as e:
        return None

def get_laptop_camera():
    """Get laptop camera stream generator"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None
        
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
        cap.release()
    except Exception as e:
        return None

def get_camera_stream():
    """
    Get camera stream with automatic fallback
    First tries ESP32-CAM, then falls back to laptop camera
    """
    print(f"Attempting to connect to ESP32-CAM at {ESP32_CAM_IP}...")
    print(f"Make sure you're connected to '{WIFI_NAME}' network")
    
    # Try ESP32-CAM first
    esp32_stream = get_esp32_stream()
    if esp32_stream is not None:
        try:
            # Test if we can get a frame
            test_frame = next(esp32_stream)
            if test_frame is not None:
                print("✓ Connected to ESP32-CAM!")
                return esp32_stream, "ESP32-CAM"
        except StopIteration:
            pass
        except Exception:
            pass
    
    # Fallback to laptop camera
    print("✗ Could not connect to ESP32-CAM")
    print("Falling back to laptop camera...")
    laptop_stream = get_laptop_camera()
    if laptop_stream is not None:
        print("✓ Using laptop camera!")
        return laptop_stream, "Laptop Camera"
    
    print("✗ Could not access any camera!")
    return None, None

