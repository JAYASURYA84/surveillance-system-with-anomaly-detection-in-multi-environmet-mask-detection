# Import the necessary packages
from tensorflow import keras
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    # Grab the dimensions of the frame and then construct a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Initialize our lists
    faces = []
    locs = []
    preds = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face ROI, preprocess it, and add it to our lists
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # Make predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# Load the face detector model
prototxtPath = r"D:\Face-Mask-Detection-master\Face-Mask-Detection-master\face_detector\deploy.prototxt"
weightsPath = r"D:\Face-Mask-Detection-master\Face-Mask-Detection-master\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the face mask detector model
maskNet = load_model("mask_detector.h5")

# Initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# Create a directory to store frames without masks if it doesn't exist
output_dir = r"D:\Face-Mask-Detection-master\Face-Mask-Detection-master\frames_without_masks"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize a counter for the frames without masks
frame_counter = 0

# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)

    # Detect faces in the frame and determine if they are wearing a face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # Loop over the detected face locations and their corresponding predictions
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"

        # If the person is not wearing a mask, save the frame to the output directory
        if label == "No Mask":
            frame_counter += 1
            filename = os.path.join(output_dir, f"frame_{frame_counter}.jpg")
            cv2.imwrite(filename, frame)

        # Draw the label and bounding box rectangle on the output frame
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
