from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('best.onnx')
cap = cv2.VideoCapture(0)

def preprocess_frame(frame):
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Preprocess frame
    processed_frame = preprocess_frame(frame)
    
    results = model(processed_frame)
    result = results[0]
    annotated_frame = result.plot() 
    cv2.imshow('YOLO Inference', annotated_frame)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()