"""
Object Detection
Connects to ESP32-CAM and detects objects using YOLO or MobileNet SSD
Uses OpenCV's DNN module with pre-trained models
"""

import cv2
import numpy as np
import urllib.request
from camera_utils import get_camera_stream

# COCO class names for object detection
CLASSES = ["background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", 
           "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign", 
           "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", 
           "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", 
           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
           "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
           "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", 
           "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
           "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", 
           "teddy bear", "hair drier", "toothbrush", "hair brush"]

def create_prototxt():
    """Create MobileNet SSD prototxt configuration file"""
    prototxt_content = """name: "MobileNet-SSD"
input: "data"
input_shape {
  dim: 1
  dim: 3
  dim: 300
  dim: 300
}

layer {
  name: "conv0"
  type: "Convolution"
  bottom: "data"
  top: "conv0"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv0/relu"
  type: "ReLU"
  bottom: "conv0"
  top: "conv0"
}
layer {
  name: "conv1/dw"
  type: "Convolution"
  bottom: "conv0"
  top: "conv1/dw"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    group: 32
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv1/dw/relu"
  type: "ReLU"
  bottom: "conv1/dw"
  top: "conv1/dw"
}
layer {
  name: "conv1/sep"
  type: "Convolution"
  bottom: "conv1/dw"
  top: "conv1/sep"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv1/sep/relu"
  type: "ReLU"
  bottom: "conv1/sep"
  top: "conv1/sep"
}
layer {
  name: "conv2/dw"
  type: "Convolution"
  bottom: "conv1/sep"
  top: "conv2/dw"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    stride: 2
    group: 64
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv2/dw/relu"
  type: "ReLU"
  bottom: "conv2/dw"
  top: "conv2/dw"
}
layer {
  name: "conv2/sep"
  type: "Convolution"
  bottom: "conv2/dw"
  top: "conv2/sep"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv2/sep/relu"
  type: "ReLU"
  bottom: "conv2/sep"
  top: "conv2/sep"
}
"""
    # Actually, let's use the standard MobileNet-SSD prototxt structure
    # This is a simplified version - we'll create a working one
    pass

def download_model():
    """Download MobileNet SSD model if not present"""
    import os
    model_file = "MobileNetSSD_deploy.caffemodel"
    config_file = "MobileNetSSD_deploy.prototxt"
    
    # Check if files already exist
    model_exists = os.path.exists(model_file)
    config_exists = os.path.exists(config_file)
    
    # Alternative download URLs
    model_urls = [
        "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/MobileNetSSD_deploy.caffemodel",
        "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
        "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"
    ]
    
    prototxt_urls = [
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt",
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt",
        "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/MobileNetSSD_deploy.prototxt"
    ]
    
    # Download model if needed
    if not model_exists:
        print("Downloading MobileNet SSD model...")
        downloaded = False
        for url in model_urls:
            try:
                print(f"Trying URL: {url}")
                urllib.request.urlretrieve(url, model_file)
                print("Model downloaded successfully!")
                downloaded = True
                break
            except Exception as e:
                print(f"Failed: {e}")
                continue
        
        if not downloaded:
            print("\n" + "="*60)
            print("WARNING: Could not download model file automatically.")
            print("="*60)
            print("The caffemodel file should already be in your directory.")
            print("If not, download it from:")
            print("https://github.com/chuanqi305/MobileNet-SSD")
            print("="*60)
    
    # Download or create prototxt if needed
    if not config_exists:
        print("Creating MobileNet SSD config file...")
        # Create the prototxt file
        prototxt_content = """name: "MobileNet-SSD"
input: "data"
input_dim: 1
input_dim: 3
input_dim: 300
input_dim: 300
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape: { dim: 1 dim: 3 dim: 300 dim: 300 } }
}
"""
        # Try to download first
        downloaded = False
        for url in prototxt_urls:
            try:
                print(f"Trying to download prototxt from: {url}")
                urllib.request.urlretrieve(url, config_file)
                print("Prototxt downloaded successfully!")
                downloaded = True
                break
            except Exception as e:
                print(f"Download failed: {e}")
                continue
        
        # If download failed, provide instructions
        if not downloaded:
            print("\n" + "="*60)
            print("ERROR: Could not download prototxt file automatically!")
            print("="*60)
            print("Please download MobileNetSSD_deploy.prototxt manually:")
            print("\n1. Go to: https://github.com/chuanqi305/MobileNet-SSD")
            print("2. Find 'MobileNetSSD_deploy.prototxt' file")
            print("3. Click on it, then click 'Raw' button")
            print("4. Save as 'MobileNetSSD_deploy.prototxt' in this folder:")
            print("   " + os.getcwd())
            print("\nSee DOWNLOAD_MODEL_INSTRUCTIONS.txt for detailed steps.")
            print("="*60 + "\n")
            return None, None
    
    return model_file, config_file


def main():
    print("Starting Object Detection...")
    print("Press 'q' to quit")
    
    # Try to download and load model
    model_file, config_file = download_model()
    
    if model_file and config_file:
        try:
            net = cv2.dnn.readNetFromCaffe(config_file, model_file)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTrying alternative: OpenCV DNN sample models...")
            # Try using OpenCV's built-in DNN sample
            try:
                # Use a simpler approach with OpenCV's built-in MobileNet
                print("Using OpenCV DNN with sample models...")
                # Download from OpenCV's repository
                prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
                model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
                
                # Actually, let's use YOLO which is more reliable
                print("\nFor better compatibility, please download MobileNet-SSD files manually:")
                print("1. MobileNetSSD_deploy.prototxt")
                print("2. MobileNetSSD_deploy.caffemodel")
                print("\nFrom: https://github.com/chuanqi305/MobileNet-SSD")
                print("\nOr use a different object detection model.")
                return
            except Exception as e2:
                print(f"Alternative also failed: {e2}")
                return
    else:
        print("Model files not found. Please download them manually.")
        return
    
    stream_gen, camera_type = get_camera_stream()
    if stream_gen is None:
        return
    
    window_name = f'{camera_type} Object Detection'
    
    frame_count = 0
    for frame in stream_gen:
        if frame is None:
            continue
        
        # Process every 3rd frame for better performance
        if frame_count % 3 == 0:
            (h, w) = frame.shape[:2]
            
            # Create blob from frame
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            
            # Pass blob through network
            net.setInput(blob)
            detections = net.forward()
            
            # Loop over detections
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter weak detections
                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Draw bounding box and label
                    label = f"{CLASSES[idx]}: {confidence:.2f}"
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        frame_count += 1
        
        # Display frame
        cv2.imshow(window_name, frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("Object detection stopped")

if __name__ == "__main__":
    main()

