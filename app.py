import pickle
from flask import Flask, render_template, request
import os
from random import random
from my_yolov6 import my_yolov6
import cv2
import urllib.request
import numpy as np
import torch
import time

yolov6_model = my_yolov6("weights/yolov6s.pt", "gpu",
                         "data/coco.yaml", 640, True)

# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

# Path to output directory
output_dir = "./output"

# Check if output directory exists, create it if not
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to process image from camera and perform object detection


def process_camera_image():
    # Read the image from the camera
    url = ' http://192.168.98.169/cam-hi.jpg'
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
        cv2.destroyAllWindows()
        return False

    return True

# Route for home page


@app.route("/", methods=['GET', 'POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
        try:
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Lưu file
                path_to_save = os.path.join(
                    app.config['UPLOAD_FOLDER'], image.filename)
                print("Save = ", path_to_save)
                image.save(path_to_save)

                # Load the saved image
                frame = cv2.imread(path_to_save)

                # Run object detection using YOLOv6
                detections = yolov6_model.infer(
                    frame, conf_thres=0.6, iou_thres=0.45)

                if len(detections) > 0:
                    # Draw bounding boxes and labels on the image
                    for detection in detections:
                        if isinstance(detection, dict):
                            label = detection['label']
                            conf = detection['conf']
                            x1, y1, x2, y2 = detection['bbox']
                            cv2.rectangle(frame, (x1, y1),
                                          (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Generate a unique filename for the annotated image
                    annotated_filename = f"annotated_{image.filename}"
                    annotated_path = os.path.join(
                        app.config['UPLOAD_FOLDER'], annotated_filename)

                    # Save the annotated image
                    cv2.imwrite(annotated_path, frame)

                    # Trả về kết quả
                    return render_template("index.html", user_image=annotated_filename, rand=str(random()),
                                           msg="Tải file lên thành công", ndet=len(detections))
                else:
                    return render_template('index.html', msg='Không nhận diện được vật thể')
            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('index.html', msg='Hãy chọn file để tải lên')

        except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html', msg='Không nhận diện được vật thể')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
