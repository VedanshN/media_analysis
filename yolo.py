from ultralytics import YOLO
from PIL import Image
import cv2 # OpenCV for drawing on the image

model = YOLO('yolov8n.pt')
image_path = 'best_frame_2.jpg' 
# --- 3. Run inference on the image ---
# The model returns a list of Results objects.
# Since we are processing one image, we get a list with one element.
results = model(image_path)
image = cv2.imread(image_path)
# The result for the first (and only) image
result = results[0]

# The 'boxes' attribute contains the detected bounding boxes
for box in result.boxes:
    # --- Get bounding box coordinates ---
    # The .xyxy attribute gives coordinates in (x1, y1, x2, y2) format
    cords = box.xyxy[0].tolist()
    x1, y1, x2, y2 = [round(x) for x in cords]
    
    # --- Get class and confidence ---
    class_id = int(box.cls[0]) # Class ID
    conf = float(box.conf[0])  # Confidence score
    class_name = model.names[class_id] # Get class name from model
    
    print(f"Detected '{class_name}' with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]")

