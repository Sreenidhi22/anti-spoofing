# train a new model
from ultralytics import YOLO

model = YOLO('../models/yolov8n.pt')

def main():
    model.train(data='Dataset/SplitData/data.yaml', epochs=75)


if __name__ == '__main__':
    main()






'''
# use to resume training if an error occurs or if training stops
from ultralytics import YOLO

# Point to your last saved weights from the restored folder
model = YOLO('runs/detect/train4/weights/last.pt')  

def main():
    model.train(resume=True)  # This tells YOLO to resume from where it left off

if __name__ == '__main__':
    main()
'''