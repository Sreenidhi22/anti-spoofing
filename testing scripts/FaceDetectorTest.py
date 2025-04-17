from cvzone.FaceDetectionModule import FaceDetector
import cv2

cap = cv2.VideoCapture(0)

detector = FaceDetector()
while True:
    success, img = cap.read()

    # Flip the image horizontally
    img = cv2.flip(img, 1)

    img, bboxs = detector.findFaces(img)

    if bboxs:
        center = bboxs[0]["center"]
        #cv2.circle(img, center, 1, (225, 0, 225), cv2.FILLED)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
