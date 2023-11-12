import os
from flask import Flask, render_template, Response, request
import cv2
import urllib.request
import numpy as np
from my_yolov6 import my_yolov6
import time

# Initialize Flask
app = Flask(__name__)

# Path to output directory
output_dir = "./output"

# Check if output directory exists, create it if not
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize YOLOv6 model
yolov6_model = my_yolov6("weights/best_ckpt.pt", "gpu",
                         "data/mydataset.yaml", 640, True)


def get_frame():
    # URL of the camera
    url = 'http://172.20.10.5/cam-hi.jpg'
    
    # Time interval for capturing images (5 seconds)
    capture_interval = 5
    last_capture_time = time.time()

    while True:
        # Read the image from the camera
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)

        # Resize the frame to 640x480
        resized_frame = cv2.resize(frame, (800, 600))

        # Run object detection using YOLOv6
        detections = yolov6_model.infer(resized_frame, conf_thres=0.6, iou_thres=0.45)

        # Draw bounding boxes and labels on the frame
        for detection in detections:
            if isinstance(detection, dict):
                label = detection['label']
                conf = detection['conf']
                x1, y1, x2, y2 = detection['bbox']
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(resized_frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Check if it's time to capture an image
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            last_capture_time = current_time
            
            # Check if there are any detections
            if len(detections) > 0:
                # Generate a unique filename for the image
                filename = os.path.join(output_dir, f"{int(time.time())}.jpg")

                # Save the image
                cv2.imwrite(filename, resized_frame)

        # Encode the frame as a JPEG image
        ret, jpeg = cv2.imencode('.jpg', resized_frame)
        frame = jpeg.tobytes()

        # Yield the frame as a byte string
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


@app.route("/", methods=['GET', 'POST'])
def home_page():
    # If it's a POST request (file upload)
    if request.method == "POST":
        try:
            # Get the uploaded file
            image = request.files['file']
            if image:
                # Save the file
                path_to_save = os.path.join(
                    app.config['UPLOAD_FOLDER'], image.filename)
                print("Save =", path_to_save)
                image.save(path_to_save)

                # Load the saved image
                frame = cv2.imread(path_to_save)
                detections, ndet = yolov6_model.infer(
                    frame, conf_thres=0.6, iou_thres=0.45)
                if ndet != 0:
                    cv2.imwrite(path_to_save, frame)

                    # Return the result
                    return render_template("index.html", user_image=image.filename, rand=str(random()),
                                           msg="File uploaded successfully", ndet=ndet)
                else:
                    return render_template('index.html', msg='No objects detected')
            else:
                # If no file is selected, prompt to upload a file
                return render_template('index.html', msg='Please select a file to upload')

        except Exception as ex:
            # If an error occurs, display the error message
            print(ex)
            return render_template('index.html', msg='Failed to detect objects')

    else:
        # If it's a GET request, display the upload interface
        return render_template('index.html')


# Route for streaming video from camera
@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

# import os
# from flask import Flask, render_template, Response, request
# import cv2
# import urllib.request
# import numpy as np
# from my_yolov6 import my_yolov6
# import time

# # Initialize Flask
# app = Flask(__name__)

# # Path to output directory
# output_dir = "./output"

# # Check if output directory exists, create it if not
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Initialize YOLOv6 model
# yolov6_model = my_yolov6("weights/best_ckpt.pt", "gpu",
#                          "data/mydataset.yaml", 640, True)


# def get_frame():
#     # URL of the camera
#     url = 'http://172.20.10.4/cam-hi.jpg'

#     while True:
#         # Read the image from the camera
#         img_resp = urllib.request.urlopen(url)
#         imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
#         frame = cv2.imdecode(imgnp, -1)

#         # Resize the frame to 640x480
#         resized_frame = cv2.resize(frame, (800, 600))

#         # Run object detection using YOLOv6
#         detections = yolov6_model.infer(
#             resized_frame, conf_thres=0.6, iou_thres=0.45)

#         # Draw bounding boxes and labels on the frame
#         for detection in detections:
#             if isinstance(detection, dict):
#                 label = detection['label']
#                 conf = detection['conf']
#                 x1, y1, x2, y2 = detection['bbox']
#                 cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(resized_frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#         #Check and save 
#         if len(detections) > 0:
#                 # Generate a unique filename for the image
#             filename = os.path.join(output_dir, f"{int(time.time())}.jpg")

#             # Save the image
#             cv2.imwrite(filename, resized_frame)

#         # Encode the frame as a JPEG image
#         ret, jpeg = cv2.imencode('.jpg', resized_frame)
#         frame = jpeg.tobytes()

#         # Yield the frame as a byte string
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/video_feed')
# def video_feed():
#     return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)


# @app.route("/", methods=['GET', 'POST'])
# def home_page():
#     # If it's a POST request (file upload)
#     if request.method == "POST":
#         try:
#             # Get the uploaded file
#             image = request.files['file']
#             if image:
#                 # Save the file
#                 path_to_save = os.path.join(
#                     app.config['UPLOAD_FOLDER'], image.filename)
#                 print("Save =", path_to_save)
#                 image.save(path_to_save)

#                 # Load the saved image
#                 frame = cv2.imread(path_to_save)
#                 detections, ndet = yolov6_model.infer(
#                     frame, conf_thres=0.6, iou_thres=0.45)
#                 if ndet != 0:
#                     cv2.imwrite(path_to_save, frame)

#                     # Return the result
#                     return render_template("index.html", user_image=image.filename, rand=str(random()),
#                                            msg="File uploaded successfully", ndet=ndet)
#                 else:
#                     return render_template('index.html', msg='No objects detected')
#             else:
#                 # If no file is selected, prompt to upload a file
#                 return render_template('index.html', msg='Please select a file to upload')

#         except Exception as ex:
#             # If an error occurs, display the error message
#             print(ex)
#             return render_template('index.html', msg='Failed to detect objects')

#     else:
#         # If it's a GET request, display the upload interface
#         return render_template('index.html')


# # Route for streaming video from camera
# @app.route('/video_feed')
# def video_feed():
#     return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)


# import os
# from flask import Flask, render_template, Response, request
# import cv2
# import urllib.request
# import numpy as np
# from my_yolov6 import my_yolov6
# import time

# # Initialize Flask
# app = Flask(__name__)

# # Path to output directory
# output_dir = "./output"
# output_labels_dir = "./output_labels"

# # Check if output directory exists, create it if not
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# if not os.path.exists(output_labels_dir):
#     os.makedirs(output_labels_dir)

# # Initialize YOLOv6 model
# yolov6_model = my_yolov6("weights/best_ckpt.pt", "gpu",
#                          "data/mydataset.yaml", 640, True)


# def get_frame():
#     # URL of the camera
#     url = 'http://192.168.168.176/cam-hi.jpg'

#     while True:
#         # Read the image from the camera
#         img_resp = urllib.request.urlopen(url)
#         imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
#         frame = cv2.imdecode(imgnp, -1)

#         # Resize the frame to 640x480
#         resized_frame = cv2.resize(frame, (800, 600))

#         # Run object detection using YOLOv6
#         detections = yolov6_model.infer(
#             resized_frame, conf_thres=0.6, iou_thres=0.45)

#         # Draw bounding boxes and labels on the frame
#         labels=[]
#         for detection in detections:
#             if isinstance(detection, dict):
#                 label = detection['label']
#                 labels.append(label)
#                 conf = detection['conf']
#                 x1, y1, x2, y2 = detection['bbox']
#                 print(labels)
#                 cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(resized_frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#         # Save labels to a text file
#         labels_filename = os.path.join(output_labels_dir, f"{int(time.time())}.txt")
#         with open(labels_filename, 'w') as f:
#             for label in labels:
#                 f.write(f"{label}\n")

#         # Encode the frame as a JPEG image
#         ret, jpeg = cv2.imencode('.jpg', resized_frame)
#         frame = jpeg.tobytes()

#         if len(detections) > 0:
#             # Generate a unique filename for the image
#             filename = os.path.join(output_dir, f"{int(time.time())}.jpg")

#             # Save the image
#             cv2.imwrite(filename, resized_frame)

#         # Yield the frame as a byte string
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/video_feed')
# def video_feed():
#     return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)


# @app.route("/", methods=['GET', 'POST'])
# def home_page():
#     # If it's a POST request (file upload)
#     if request.method == "POST":
#         try:
#             # Get the uploaded file
#             image = request.files['file']
#             if image:
#                 # Save the file
#                 path_to_save = os.path.join(
#                     app.config['UPLOAD_FOLDER'], image.filename)
#                 print("Save =", path_to_save)
#                 image.save(path_to_save)

#                 # Load the saved image
#                 frame = cv2.imread(path_to_save)
#                 detections, ndet = yolov6_model.infer(
#                     frame, conf_thres=0.6, iou_thres=0.45)
#                 if ndet != 0:
#                     cv2.imwrite(path_to_save, frame)

#                     # Return the result
#                     return render_template("index.html", user_image=image.filename, rand=str(random()),
#                                            msg="File uploaded successfully", ndet=ndet)
#                 else:
#                     return render_template('index.html', msg='No objects detected')
#             else:
#                 # If no file is selected, prompt to upload a file
#                 return render_template('index.html', msg='Please select a file to upload')

#         except Exception as ex:
#             # If an error occurs, display the error message
#             print(ex)
#             return render_template('index.html', msg='Failed to detect objects')

#     else:
#         # If it's a GET request, display the upload interface
#         return render_template('index.html')


# # Route for streaming video from camera
# @app.route('/video_feed')
# def video_feed():
#     return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)

