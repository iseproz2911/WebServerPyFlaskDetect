import urllib.request
import cv2
import numpy as np
import os
import torch
from my_yolov6 import my_yolov6
import time

# URL of the camera
url = 'http://192.168.1.78/cam-hi.jpg'

# Initialize YOLOv6 model
yolov6_model = my_yolov6("./weights/yolov6s.pt", "gpu",
                         "./data/coco.yaml", 640, True)

# Path to output directory
output_dir = "./output"

# Check if output directory exists, create it if not
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

while True:
    # Read the image from the camera
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgnp, -1)

    # Resize the frame to 640x480
    resized_frame = cv2.resize(frame, (640, 480))

    # Run object detection using YOLOv6
    detections = yolov6_model.infer(
        resized_frame, conf_thres=0.6, iou_thres=0.45)

    # Draw bounding boxes and labels on the frame
    for detection in detections:
        if isinstance(detection, dict):
            label = detection['label']
            conf = detection['conf']
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Camera', resized_frame)

    # Save image if there is any detection
    if len(detections) > 0:
        # Generate a unique filename for the image
        filename = os.path.join(output_dir, f"{int(time.time())}.jpg")

        # Save the image
        cv2.imwrite(filename, resized_frame)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cv2.destroyAllWindows()



# Function to process image from camera and perform object detection


# def process_camera_image():
#     # URL of the camera
#     url = 'http://192.168.150.205/cam-hi.jpg'

#     # Read the image from the camera
#     img_resp = urllib.request.urlopen(url)
#     imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
#     frame = cv2.imdecode(imgnp, -1)

#     # Resize the frame to 640x480
#     resized_frame = cv2.resize(frame, (640, 480))

#     # Run object detection using YOLOv6
#     detections, ndet = yolov6_model.infer(
#         resized_frame, conf_thres=0.6, iou_thres=0.45)

#     # Draw bounding boxes and labels on the frame
#     for detection in detections:
#         if isinstance(detection, dict):
#             label = detection['label']
#             conf = detection['conf']
#             x1, y1, x2, y2 = detection['bbox']
#             cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(resized_frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Save image if there is any detection
#     if ndet > 0:
#         # Generate a unique filename for the image
#         filename = os.path.join(output_dir, f"{int(time.time())}.jpg")

#         # Save the image
#         cv2.imwrite(filename, resized_frame)

#     # Convert the frame to JPEG format for streaming
#     ret, jpeg_frame = cv2.imencode('.jpg', resized_frame)
#     return jpeg_frame.tobytes()
