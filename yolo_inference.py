from ultralytics import YOLO

# call YOLO
model = YOLO('yolov8x.pt')
result = model.track('input_videos/input_video.mp4',conf=0.2, save=True) 
# print(result)
# print("boxes:")
# for box in result[0].boxes: # see the bounding box 
#     print(box)