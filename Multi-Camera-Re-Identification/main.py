import os
import random

import cv2
from ultralytics import YOLO

from deep_sort_realtime.deepsort_tracker import DeepSort

video_path = os.path.join('.', 'people.mp4')
video_out_path = os.path.join('.', 'out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("best.pt")

tracker = DeepSort(max_age = 5)

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5
while ret:

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append(([x1, y1, x2, y2] , score , class_id))
        

        tracks = tracker.update_tracks(detections , frame = frame)


        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id

            ltrb = track.to_ltwh()

            x1 , y1 , w1 , h1 = map(int , ltrb)  
            track_id = int(track_id)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1) + int(w1) , int(y2) + int(h1)), (colors[track_id % len(colors)]), 3)
        


        """cv2.imshow('frame' , frame)
        if cv2.waitKey(1) == ord('q'):
            break"""

    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()
