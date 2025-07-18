from ultralytics import YOLO
import shutil

model = YOLO("best.pt")
model.export(format="engine")

tensorrt_model = YOLO("best.engine")

source = "best.onnx"
destination = "cpp"

if tensorrt_model:
    shutil.move(source,destination)
    print("Has moved ONNX model to folder!")

