from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

model = YOLO("yolov8n.pt")
model = YOLO("yolov8n.yaml").load("yolov8n.pt")

resaults = model.train(data="dataset/Human_database/data.yaml",epochs=100,imgsz=320,batch=8,workers=0)
