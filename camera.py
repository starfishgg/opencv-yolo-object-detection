# This code captures video from the webcam and displays it in a window.
# It uses the OpenCV library to access the webcam and display the video feed.
# Press the Esc key to exit the video feed.
# Note: Make sure you have OpenCV installed in your Python environment to run this code.
# Uses a pre-built Haar Cascade (built into OpenCV) to detect faces in the video feed and draws rectangles around them.

# Expect some lag as we are not multi-threading camera capture, YOLO detection and rendering

import os
import cv2
import numpy as np
import time

# Load YOLO or YOLO-tiny model and classes
MODEL = "full"  # "tiny" or "full"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if MODEL == "tiny":
    weights = os.path.join(BASE_DIR, "models", "yolov3-tiny.weights")
    config = os.path.join(BASE_DIR, "models", "yolov3-tiny.cfg")
else:
    weights = os.path.join(BASE_DIR, "models", "yolov3.weights")
    config = os.path.join(BASE_DIR, "models", "yolov3.cfg")

names = os.path.join(BASE_DIR, "models", "coco.names")

net = cv2.dnn.readNet(weights, config)
with open(names, "r") as f:
        classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Capture video from the webcam
cap = cv2.VideoCapture(0) # 0 = webcam
# cap = cv2.VideoCapture("videos/demo.mp4")

# track the frame number, we won't run detection on every frame to improve performance
# other options:
#       use YOLOv2-tiny, but slightly less accurate
#       reside the frame before feeding it to YOLO, this will also cause some loss of accuracy
#       run on GPU, probably 10-30x faster, but requires installing OpenCV with CUDA support, which can be a bit tricky


frame_id = 0
skip_frames = 2

class_ids, confidences, boxes = [], [], []

start = time.time()

while True:
        ret, frame = cap.read()
        if not ret:
                break

        frame_id += 1

        # Some test code
        print("FRAME:", frame_id)
        height, width, channels = frame.shape

        # Only run detection every N frames to increase performance
        if frame_id % skip_frames == 0:

                print("RUNNING YOLO")
                # Detect objects, 320x320 resolution is used for faster processing, adjust as needed
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                class_ids, confidences, boxes = [], [], []

                for out in outs:
                        for detection in out:
                                scores = detection[5:]
                                class_id = np.argmax(scores)
                                confidence = scores[class_id]

                                if confidence > 0.5:
                                        # Object detected
                                        center_x = int(detection[0] * width)
                                        center_y = int(detection[1] * height)
                                        w = int(detection[2] * width)
                                        h = int(detection[3] * height)

                                        x = int(center_x - w/2)
                                        y = int(center_y - h/2)

                                        boxes.append([x,y,w,h])
                                        confidences.append(float(confidence))
                                        class_ids.append(class_id)

                # Non-max suppression to remove overlapping boxes
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw boxes every frame so they don't flicker
        if len(boxes) > 0 and len(indexes) > 0:
                for i in indexes:
                        # Handle the indexes in a version-agonistic way.
                        # This new code should now work whether we return [0,2,4], [[9],[2],[4]], (0,) or a NumPy array, we won't just .flatten it anymore as you can't do that to a tuple object
                        if isinstance(i, (list, tuple, np.ndarray)):
                                i = i[0]

                        x, y, w, h = boxes[i]

                        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Display the resulting frame
        cv2.imshow("YOLO Object Detection", frame)

        # press Esc to exit
        if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()
