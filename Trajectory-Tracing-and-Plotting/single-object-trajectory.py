import cv2
import matplotlib.pyplot as plt

# Open a video file or use a camera feed
cap = cv2.VideoCapture(0)  # Replace 'video.mp4' with your video file or 0 for the default camera
# Initialize the tracker
tracker = cv2.TrackerCSRT_create()  # You can use different trackers based on your needs

# Read the first frame
ret, frame = cap.read()

# Select the object to track (e.g., draw a bounding box around it)
bbox = cv2.selectROI(frame, False)

# Initialize the tracker with the selected region
tracker.init(frame, bbox)

# Initialize an empty list to store the trajectory points
trajectory = []

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker
    success, bbox = tracker.update(frame)

    # If tracking is successful, draw the bounding box and update the trajectory
    if success:
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Add the object's center to the trajectory
        center_x = x + w // 2
        center_y = y + h // 2
        trajectory.append((center_x, center_y))

        # Draw lines connecting consecutive trajectory points
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Object Tracking', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

# Extract x and y coordinates from the trajectory list
x_coordinates, y_coordinates = zip(*trajectory)

# Create a Matplotlib scatter plot for the trajectory
plt.figure(figsize=(8, 6))
plt.scatter(x_coordinates, y_coordinates, c='blue', label='Object Trajectory')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Object Trajectory')
plt.legend()
plt.grid(True)

# Show the Matplotlib plot
plt.show()
