from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model = YOLO("best.pt")

val_result = model.val()
print(val_result.box.maps)
