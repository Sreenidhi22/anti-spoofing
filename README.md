# AntiSpoofing

This Python-based real-time face classification system leverages a YOLOv8 deep learning model to distinguish between "real" and "fake" faces using webcam input. The project is built using OpenCV for image processing, Ultralytics YOLO for object detection, and the cvzone library for enhanced visual rendering of bounding boxes and text overlays.

Upon launching the script, the webcam feed is captured, flipped horizontally for a mirror-like interface, and processed frame-by-frame. The YOLOv8 model (n_version_4_75.pt) is used to analyze each frame and detect faces. Once a face is detected, the model classifies it into one of two categories: "real" or "fake". Only detections above a defined confidence threshold (default is 0.6) are considered for visualization and feedback.

Bounding boxes are drawn around detected faces using cvzone.cornerRect, with color-coded indicators—green for real faces and red for fake ones. Confidence scores are displayed alongside the label, providing the user with real-time feedback about the model’s predictions. The class names are dynamically retrieved and displayed with each detection.

Additionally, the script calculates and displays the frames per second (FPS), offering insights into system performance and responsiveness. This metric is particularly useful for monitoring the efficiency of the detection pipeline and ensuring the system runs smoothly on available hardware.

The application can serve as a standalone module or be integrated into larger security or biometric systems, such as face-based attendance systems or access control applications, where it’s critical to verify face authenticity. It can also assist in preventing spoofing attacks by flagging fraudulent face presentations in real-time.

Built to be lightweight and responsive, this system is ideal for edge deployment, educational use, or as a prototype for more advanced biometric security applications. It provides an interactive demonstration of AI in computer vision, making it suitable for researchers, developers, and enthusiasts interested in face detection and anti-spoofing technologies.

To exit the system, users can press the 'q' key, at which point the webcam is released and all OpenCV windows are closed gracefully.



Required installation:
```bash
pip install mediapipe ultralytics cvzone
```
Also to run these codes with better effieiency from CPU to run on GPU, PyTorch locally can be used, check your compatibility and specifications and pip install it to the virtual environment: https://pytorch.org/get-started/locally/

Create the Dataset folder too, with relavent folders shown inside !!

File structure for reference:


![image](https://github.com/user-attachments/assets/49e432bb-ff9e-4c38-b6b4-9be2047fce88)








