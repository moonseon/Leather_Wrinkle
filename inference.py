from ultralytics import YOLO
model = YOLO("yolov8n.pt")
def detect(img):
    return model.predict(img)
