from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="engine")

tensorrt_model = YOLO("best.engine")

results = tensorrt.model("https://ultralytics.com/images/bus.jpg")
