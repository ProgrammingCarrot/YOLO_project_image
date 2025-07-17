import pyrealsense2 as rs
from ultralytics import YOLO
import cv2
import numpy as np

# get camera pipeline
pipeline = rs.pipeline()
config = rs.config()

# get device information
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_lines = str(device.get_info(rs.camera_info.product_line))

# Detection if Color Sensor Exist
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        print("found Color Sensor!")
        break

if not found_rgb:
    print("Required Color Sensor!")
    exit(0)

config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)


# Import Model
def model(engine_name,pt_name):
    try:
        model = YOLO(engine_name)
    except Exception as e:
        print(f"Error:{e}")
        print("TensorRT model ins not exist,Attempting to load Torch model")
        model = YOLO(pt_name)
    finally:
        return model

try:
    model = model("best.engine","best.pt")
    profile = pipeline.start(config)
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("wait for frames")
            continue

        color_image = np.asanyarray(color_frame.get_data())
        result = model(color_image,stream=False,verbose=False)
        print(result[0])

        detection_image = result[0].plot()
        detection_image = cv2.cvtColor(detection_image,cv2.COLOR_RGB2BGR)
        cv2.namedWindow("Result",cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Result",detection_image)

        keys = cv2.waitKey(1)
        if keys == 27:
            break
finally:
    print("End")
    cv2.destroyAllWindows()
    pipeline.stop()
