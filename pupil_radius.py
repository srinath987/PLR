import cv2
import time
import matplotlib.pyplot as plt

# Open the default camera
cap = cv2.VideoCapture(0)

# Initialize variables for time and radius
start_time = time.time()
radius_values = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect the face or eye region using a cascade classifier
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
    faces = face_cascade.detectMultiScale(gray_blur, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray_blur, 1.3, 5)

    # Get the eye region
    for (ex, ey, ew, eh) in eyes:
        eye_roi = gray_blur[ey:ey+eh, ex:ex+ew]

        # Apply thresholding to separate the pupil from the iris
        _, threshold = cv2.threshold(eye_roi, 40, 255, cv2.THRESH_BINARY_INV)

        # Find the contours of the thresholded image
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        # Find the contour with the largest area, which corresponds to the pupil
        pupil_contour = max(contours, key=cv2.contourArea)

        # Calculate the centroid of the pupil contour
        moments = cv2.moments(pupil_contour)
        if moments['m00'] == 0:
            continue
        
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # Calculate the radius of the pupil
        radius = int(cv2.minEnclosingCircle(pupil_contour)[1])
        radius_values.append(radius)

        # Draw the circle around the pupil on the original image
        cv2.circle(frame, (ex+cx, ey+cy), radius, (0, 255, 0), 2)

    # Display the image with the circle around the pupil
    cv2.imshow('Eye Image', frame)

    # Exit the loop after 10 seconds
    elapsed_time = time.time() - start_time
    if elapsed_time >= 10:
        break

    # Wait for key press to continue
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Plot the radius over time
time_values = [i/30 for i in range(len(radius_values))] # assuming 30 fps
plt.plot(time_values, radius_values)
plt.xlabel('Time (seconds)')
plt.ylabel('Radius')
plt.show()
