import math
import time
import cv2
import cvzone
from ultralytics import YOLO

# Set confidence threshold for object detection
confidence = 0.6

# Initialize video capture from the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)  # Set the width of the frame
cap.set(4, 480)  # Set the height of the frame

# Load the YOLO model for detecting objects (path to the trained model)
model = YOLO("../models/n_version_4_75.pt")

# Define the class names for detection ("fake" and "real")
classNames = ["fake", "real"]

# Initialize variables for calculating frames per second (FPS)
prev_frame_time = 0
new_frame_time = 0

# Start the webcam feed and process frames
while True:
    # Capture the new frame and calculate FPS
    new_frame_time = time.time()
    success, img = cap.read()

    if not success:  # If there is an issue with reading the frame, continue to the next iteration
        continue

    # Flip the image horizontally (for a mirror-like effect)
    img = cv2.flip(img, 1)

    # Perform object detection using the YOLO model on the current frame
    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes  # Get bounding boxes for detected objects
        for box in boxes:
            # Extract coordinates of the bounding box (top-left and bottom-right)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Calculate width and height of the bounding box
            w, h = x2 - x1, y2 - y1

            # Get the confidence score of the detection
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Get the class of the detected object (0 for fake, 1 for real)
            cls = int(box.cls[0])

            if conf > confidence:  # Only consider detections above the confidence threshold
                # Set color for bounding box and label based on class
                color = (0, 255, 0) if classNames[cls] == 'real' else (0, 0, 255)

                # Draw a corner rectangle around the detected object
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)

                # Display the class name and confidence on the frame
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color, colorB=color)

    # Calculate and print FPS (frames per second)
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps:.2f}")

    # Show the processed image with bounding boxes and labels
    cv2.imshow("Image", img)

    # Wait for the 'q' key to be pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows when done
cap.release()
cv2.destroyAllWindows()
