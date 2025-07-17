from ultralytics import YOLO
import cv2

try:
    model = YOLO("yolo11n.engine")
except Exception as e:
    print(f"Error:{e}")
    print(".engine is not exsiting.Attempting to load .pt")
    model = YOLO("yolo11n.engine")

result = model("https://ultralytics.com/images/bus.jpg")

frame = result[0].plot()
bgr_image = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

print("start to print image")
try:
    while True:
        image = cv2.imread("bus.jpg")
        cv2.namedWindow("Result",cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Result",bgr_image)
        keys = cv2.waitKey(0)
        if keys == 27:
            break
finally:
    print("The End")
    cv2.destroyAllWindows()
