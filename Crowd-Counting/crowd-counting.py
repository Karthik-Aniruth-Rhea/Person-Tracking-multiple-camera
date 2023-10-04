# Import YOLOv5 and related dependencies
from pathlib import Path
import torch
import cv2
from tracker import *

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set the video file path
video_path = 'people-walking.mp4'

# Open the video capture
cap = cv2.VideoCapture(video_path)

# Initialize the tracker
tracker = Tracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection with YOLOv5 on the frame
    results = model(frame)

    # Extract bounding boxes and labels from the YOLOv5 results
    objects_rect = []
    for pred in results.pred:
        for det in pred:
            x, y, w, h, score, class_id = det.tolist()
            if class_id == 0:  # Assuming class 0 corresponds to people
                objects_rect.append((int(x), int(y), int(w * 0.1), int(h * 0.2)))

    # Update the tracker with the detected bounding boxes
    objects_bbs_ids = tracker.update(objects_rect)

    # Calculate the count of people detected in the current frame
    current_frame_count = len(objects_bbs_ids)

    # Visualize the tracked objects by drawing bounding boxes and IDs
    for x, y, w, h, id in objects_bbs_ids:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw the count of people detected in the current frame
    cv2.putText(frame, f"Current Frame Count: {current_frame_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('FRAME', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
