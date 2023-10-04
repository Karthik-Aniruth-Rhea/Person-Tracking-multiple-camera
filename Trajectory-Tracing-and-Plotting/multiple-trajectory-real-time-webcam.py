import cv2
import matplotlib.pyplot as plt

# Open a video file or use a camera feed
cap = cv2.VideoCapture(0)

# Initialize a list to store trackers for multiple objects
trackers = []
object_trajectories = []  # List to store trajectories for each object

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally to match your movement
    frame = cv2.flip(frame, 1)

    # Loop through the objects to track (you can add more objects as needed)
    for tracker in trackers:
        # Update the tracker
        success, bbox = tracker.update(frame)

        # If tracking is successful, draw the bounding box and update the trajectory
        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add the object's center to the trajectory with inverted y-coordinate
            center_x = x + w // 2
            center_y = y + h // 2
            object_trajectories[trackers.index(tracker)].append((center_x, center_y))

            # Draw lines connecting consecutive trajectory points
            for i in range(1, len(object_trajectories[trackers.index(tracker)])):
                cv2.line(frame, object_trajectories[trackers.index(tracker)][i - 1],
                         object_trajectories[trackers.index(tracker)][i], (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Object Tracking', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Add new objects to track (you can add more objects as needed)
    if len(trackers) < 2:  # Track up to 2 objects
        bbox = cv2.selectROI(frame, False)
        tracker = cv2.TrackerMIL_create()  # Use MIL tracker
        tracker.init(frame, bbox)
        trackers.append(tracker)
        object_trajectories.append([])  # Initialize an empty trajectory for the new object

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

# Create a Matplotlib plot for the object trajectories
plt.figure(figsize=(8, 6))

# Loop through the trajectories and create a scatter plot for each object
for idx, trajectory in enumerate(object_trajectories):
    if trajectory:  # Check if there are valid trajectories to plot
        x_coordinates, y_coordinates = zip(*trajectory)
        x = [-x for x in x_coordinates]
        y = [-y for y in y_coordinates]
        plt.plot(x, y, label=f'Object {idx + 1} Trajectory')

plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Object Trajectories')
plt.legend()
plt.grid(True)

# Save the Matplotlib plot as an image in the same directory
plt.savefig('object_trajectories_plot.png')

# Show the Matplotlib plot (optional)
plt.show()
