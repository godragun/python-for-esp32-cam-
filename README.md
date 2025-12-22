<p align="center">
  <img src="https://raw.githubusercontent.com/godragun/python-for-esp32-cam-/main/pythonesp32.jpeg" alt="Python ESP32" width="360" style="margin-right:12px;"/>
  <img src="https://raw.githubusercontent.com/godragun/python-for-esp32-cam-/main/puthonesp32cam.jpeg" alt="Put Phone ESP32 CAM" width="360" style="margin-left:12px;"/>
</p>

# ESP32-CAM Python Applications

This repository contains multiple Python applications that connect to an ESP32-CAM via Wi-Fi and perform various computer vision and gesture recognition tasks.

## Camera Setup

- **Wi-Fi Network**: "harendra cam free wifi"
- **IP Address**: 192.168.4.1
- **Stream URL**: http://192.168.4.1/stream

## Installation

1. Install Python 3.8 or higher
2. Install required packages:

```bash
pip install -r requirements.txt
```

**Note:** The installation may take several minutes as it downloads large packages like TensorFlow and PyTorch (required for emotion recognition). Please be patient and let the installation complete.

If you encounter issues with `fer` module:
```bash
python -m pip install fer
```

## Applications

### 1. Face Recognition (`face_recognition.py`)
- Detects faces in the camera feed
- Colors change based on number of faces detected:
  - Purple: 1 face
  - Green: 2 faces
  - Yellow: 3 faces
  - Red: 4+ faces

### 2. Hand Detection (`hand_detection.py`)
- Detects hands and displays all 21 hand landmarks
- Shows hand connections and tracking points

### 3. Face Emotion Recognition (`face_emotion.py`)
- Recognizes emotions from faces (happy, sad, angry, etc.)
- Displays emotion labels and confidence scores
- Requires FER library

### 4. Object Detection (`object_detection.py`)
- Detects objects using MobileNet SSD model
- Identifies 80+ different object classes
- Automatically downloads model files on first run

### 5. Red Dot Game (`red_dot_game.py`)
- Interactive game where you tap red dots with your finger
- Score tracking
- Hand gesture-based interaction

### 6. Brightness Control (`brightness_control.py`)
- Controls screen brightness using hand gestures
- Thumbs up: Increase brightness
- Thumbs down: Decrease brightness

### 7. Volume Control (`volume_control.py`)
- Controls system volume using hand gestures
- Thumbs up: Increase volume
- Thumbs down: Decrease volume
- Windows only (requires pycaw)

### 8. Chrome Opener (`open_chrome.py`)
- Opens Google Chrome browser when 5 fingers are detected
- Gesture-based browser launcher

### 9. Finger Counter (`finger_count.py`)
- Counts the number of extended fingers
- Supports detection of both hands
- Shows which fingers are extended

## Usage

1. Connect to the ESP32-CAM Wi-Fi network: "harendra cam free wifi" (or use laptop camera as fallback)
2. Run any of the Python scripts:

```bash
python face_recognition.py
python hand_detection.py
python face_emotion.py
python object_detection.py
python red_dot_game.py
python brightness_control.py
python volume_control.py
python open_chrome.py
python finger_count.py
```

3. Press 'q' to quit any application

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe (for hand detection)
- FER (for emotion recognition)
- TensorFlow (for emotion recognition)
- screen-brightness-control (for brightness control)
- pycaw (for volume control on Windows)

## Notes

- Make sure you're connected to the ESP32-CAM Wi-Fi network before running any script
- Some applications may require internet connection to download models on first run
- Volume and brightness control may require administrator privileges on some systems
- Chrome opener may need path adjustment based on your Chrome installation location
