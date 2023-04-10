import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Initialize the arrays to store data
times = []
radii = []

# Get the start time
start_time = time.time()

# Capture the video for 10 seconds and store the frames in an array
frames = []
while(time.time() - start_time < 10):
    ret, frame = cap.read()
    frames.append(frame)

# Process the frames and store the radii in an array
for frame in frames:
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect the eye region using a cascade classifier
    eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray_blur, 1.3, 5)

    # Get the eye region and calculate the radius of the pupil
    for (ex, ey, ew, eh) in eyes:
        eye_roi = gray_blur[ey:ey+eh, ex:ex+ew]

        # Apply thresholding to separate the pupil from the iris
        _, threshold = cv2.threshold(eye_roi, 40, 255, cv2.THRESH_BINARY_INV)
        
        # Find the contours of the thresholded image
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area, which corresponds to the pupil
        if len(contours) > 0:
            pupil_contour = max(contours, key=cv2.contourArea)
            
            # Calculate the radius of the pupil
            radius = int(cv2.minEnclosingCircle(pupil_contour)[1])

            # Append the current time and radius to the arrays
            current_time = time.time() - start_time
            times.append(current_time)
            radii.append(radius)

            # Draw the circle around the pupil on the original image
            cv2.circle(frame, (ex+int(pupil_contour[:, 0, 0].mean()), ey+int(pupil_contour[:, 0, 1].mean())), radius, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

# Release the resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Plot the radius against time graph
plt.plot(times, radii)
plt.xlabel('Time (s)')
plt.ylabel('Radius')
plt.show()
