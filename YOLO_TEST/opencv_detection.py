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
        model = YOLO(pt_name,task="detect")
    finally:
        return model

# Get Class Color
def getColors(cls_num):
    base_color = [(255,0,0),(0,255,0),(0,0,255)]
    color_index = cls_num % len(base_color)
    increments = [(1,-2,-1),(-2,1,-1),(-1,-1,2)]
    color = [base_color[color_index][i] + increments[color_index][i] * 
            (cls_num // len(base_color)) % 256 for i in range(3)]
    return tuple(color)

try:
    model = model("best.onnx","best.pt") # use clean-suit detection model
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("wait for frames")
            continue

        color_image = np.asanyarray(color_frame.get_data())
        result = model(color_image,stream=False,verbose=False)

        # Get detect result and print on the screen
        class_names = result[0].names
        for box in result[0].boxes:
            if box.conf[0] > 0.5:
                [x1,y1,x2,y2] = box.xyxy[0]
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                cls = int(box.cls[0])
                class_name = class_names[cls]
                color = getColors(cls)
                
                distance = depth_frame.get_distance((x1+x2)//2,(y1+y2)//2)

                if abs(y2-y1) > color_frame.get_height()/2:
                    result_label = (x1+30,y1+30)
                else:
                    result_label = (x1,y1)

                cv2.rectangle(color_image,(x1,y1),(x2,y2),color,3)
                cv2.putText(color_image,f'{class_name},{distance:.2f}m \n {box.conf[0]}',
                            result_label,cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)


        cv2.namedWindow("Result",cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Result",color_image)

        keys = cv2.waitKey(1)
        if keys == 27:
            break
finally:
    print("End")
    cv2.destroyAllWindows()
    pipeline.stop()
