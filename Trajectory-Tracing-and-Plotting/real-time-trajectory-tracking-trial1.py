import cv2
import numpy as np
import time
import torch
from utils.general import non_max_suppression, scale_boxes

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    if color is None:
        color = (0, 255, 0)  # Default color is green (BGR format)

    if line_thickness is None:
        line_thickness = max(int(min(img.shape[0:2]) / 200), 1)  # Adjust line thickness based on image size

    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
    if label:
        tf = max(line_thickness - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# Load the YOLOv5 model
#model_path = 'yolov5s.pt'
# #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = attempt_load(model_path, device=device)
#model.eval()  # Set the model to evaluation mode

tracker = cv2.selectROI(frame, False)

# Function to detect people using YOLOv5
def detect_people(frame):
    # Convert the frame to a PyTorch tensor
    img0 = torch.from_numpy(frame).to(device).float()  # Convert to float data type

    # Perform any necessary preprocessing here (e.g., resizing, normalization)

    # Ensure that the input tensor has the expected shape (e.g., NCHW format)
    img0 = img0.permute(2, 0, 1).unsqueeze(0)  # Adjust the dimensions as needed

    # Perform inference
    results = model(img0)


    # Post-process the results to get bounding boxes
    results = non_max_suppression(results, 0.5, 0.5)

    people = []
    for pred in results:
        if pred is not None and len(pred):
            # Scale the coordinates to match the frame size
            pred = scale_boxes(img0.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                if cls == 0:  # Class ID 0 represents 'person'
                    x, y, x2, y2 = xyxy
                    people.append((int(x), int(y), int(x2), int(y2)))

    return people

# Rest of the code (tracking, trajectory, speed calculation) remains the same

# Open a video file or use a camera feed
cap = cv2.VideoCapture('people-walking.mp4')  # Replace 'your_video.mp4' with your video file

# Initialize empty lists to store trajectory points, timestamps, and speeds
trajectory = []
timestamps = []
speeds = []

# Rest of the code (tracking, trajectory, speed calculation) remains the same
while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally to match your movement
    frame = cv2.flip(frame, 1)

    # Record the timestamp
    timestamp = time.time()

    # Detect people in the frame
    people = detect_people(frame)

    # Update the trajectory for each detected person
    for person in people:
        x, y, x2, y2 = person
        center_x = (x + x2) // 2
        center_y = (y + y2) // 2
        trajectory.append((center_x, center_y))
        timestamps.append(timestamp)

    # Calculate the average speed of the crowd
    if len(trajectory) >= 2:
        dx = [trajectory[i][0] - trajectory[i - 1][0] for i in range(1, len(trajectory))]
        dy = [trajectory[i][1] - trajectory[i - 1][1] for i in range(1, len(trajectory))]
        delta_t = [timestamps[i] - timestamps[i - 1] for i in range(1, len(trajectory))]

        speeds = [np.sqrt(dx[i] ** 2 + dy[i] ** 2) / delta_t[i] for i in range(len(delta_t))]
        average_speed = sum(speeds) / len(speeds)
        print("Average Speed: {:.2f} pixels per second".format(average_speed))

    # Draw people bounding boxes and display the frame
    for person in people:
        x, y, x2, y2 = person
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('People Tracking', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()