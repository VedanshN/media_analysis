import cv2
import numpy as np
'''
A WIP script to create a simple timeline visualization using OpenCV.
'''
# Create a blank white image
width, height = 800, 400
image = np.ones((height, width, 3), dtype="uint8") * 255

# Timeline parameters
start_x = 100
end_x = 700
y = 200
num_events = 5
event_spacing = (end_x - start_x) // (num_events - 1)

# Draw the timeline
cv2.line(image, (start_x, y), (end_x, y), (0, 0, 0), 2)

# Add events to the timeline
events = ["Start", "Event 1", "Event 2", "Event 3", "End"]
for i, event in enumerate(events):
    x = start_x + i * event_spacing
    cv2.circle(image, (x, y), 10, (0, 0, 255), -1)  # Draw event point
    cv2.putText(image, event, (x - 30, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Display the timeline
cv2.imshow("Timeline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
