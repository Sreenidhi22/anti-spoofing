import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from time import time

# parameters ################

classID = 0  # 0 is fake and 1 is real (for the classification of the face)
outputFolderPath = 'Dataset/DataCollect'  # Path to save collected data (images and labels)
confidence = 0.8  # Confidence threshold to consider a face as valid
save = True  # Flag to indicate whether to save the images and labels or not
blueThreshold = 40  # Threshold for the blur detection (Laplacian variance)

debug = False  # Flag for enabling debug mode (will show the original image with additional information)
offsetPercentageW = 10  # Percentage of offset for width
offsetPercentageH = 20  # Percentage of offset for height
canWidth, camHeight = 640, 480  # Resolution of the webcam capture
floatingPoint = 6  # Number of decimal places for normalized values

##############################


cap = cv2.VideoCapture(0)  # Open webcam (0 means default camera)
cap.set(3, canWidth)  # Set the width of the capture frame
cap.set(4, camHeight)  # Set the height of the capture frame
detector = FaceDetector()  # Initialize the face detector

while True:
    success, img = cap.read()  # Read the current frame from the webcam
    img = cv2.flip(img, 1)  # Flip the image horizontally (mirror effect)
    imgOut = img.copy()  # Create a copy of the original image to draw on

    img, bboxs = detector.findFaces(img, draw=False)  # Detect faces in the image (no drawing on the image yet)

    listBlur = []  # List to track if faces are blurry (True for blurry, False for clear)
    listInfo = []  # List to store normalized face information (for label txt files)

    if bboxs:  # If faces are detected
        for bbox in bboxs:  # Iterate through each detected face
            x, y, w, h = bbox["bbox"]  # Get bounding box coordinates of the face (x, y, w, h)
            score = bbox["score"][0]  # Get the confidence score of the detected face

            # Check if the score is above the confidence threshold to consider it as a valid face
            if score > confidence:

                # Apply offset to the bounding box to give more room around the face
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)  # Adjust x coordinate with offset
                w = int(w + offsetW * 2)  # Adjust width with offset
                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)  # Adjust y coordinate with offset
                h = int(h + offsetH * 3.5)  # Adjust height with offset

                # Ensure that the bounding box values are not below 0
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                # Crop the face region from the image
                imgFace = img[y:y + h, x:x + w]
                # Calculate the blurriness of the face region using the Laplacian variance
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue > blueThreshold:  # If the face is clear
                    listBlur.append(True)
                else:  # If the face is blurry
                    listBlur.append(False)

                # Normalize the bounding box values for the label file
                ih, iw, _ = img.shape  # Get the height and width of the image
                xc, yc = x + w / 2, y + h / 2  # Calculate the center of the face (x, y)

                # Normalize the x, y, width, and height to [0, 1] range
                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)

                # Ensure that the normalized values are not greater than 1
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                # Store the normalized values for the label text file
                listInfo.append(f'{classID} {xcn} {ycn} {wn} {hn}\n')

                # Draw the bounding box and the score/blurriness text on the image
                cv2.rectangle(imgOut, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score:{int(score * 100)}% Blur: {blurValue}', (x, y - 20), scale=2,
                                   thickness=2)

                # If debug is enabled, show the original image with bounding box and text as well
                if debug:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'Score:{int(score * 100)}% Blur: {blurValue}', (x, y - 20), scale=2,
                                       thickness=2)

        # If the 'save' flag is enabled, save the images and labels
        if save:
            # If all faces are clear (not blurry)
            if all(listBlur) and listBlur != []:
                # Save the current image with a unique filename based on the current time
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0] + timeNow[1]  # Remove the decimal part of the time
                cv2.imwrite(f'{outputFolderPath}/{timeNow}.jpg', img)

                # Save the label file with the normalized values for each face detected
                for info in listInfo:
                    f = open(f'{outputFolderPath}/{timeNow}.txt', 'a')
                    f.write(info)
                    f.close()

    # Display the processed image (with bounding boxes and labels) on the screen
    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)  # Wait for 1 ms for key press
