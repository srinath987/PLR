import cv2

# Create a VideoCapture object to read from the camera
cap = cv2.VideoCapture(0)

# Load the cascade classifiers
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

# Loop over the frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect the face and eye regions in the frame
        faces = face_cascade.detectMultiScale(gray_blur, 1.3, 5)
        eyes = eye_cascade.detectMultiScale(gray_blur, 1.3, 5)

        # Get the eye region
        for (ex, ey, ew, eh) in eyes:
            eye_roi = gray_blur[ey:ey+eh, ex:ex+ew]

            # Apply thresholding to separate the pupil from the iris
            _, threshold = cv2.threshold(eye_roi, 40, 255, cv2.THRESH_BINARY_INV)

            # Find the contours of the thresholded image
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Find the contour with the largest area, which corresponds to the pupil
            if len(contours) > 0:
                pupil_contour = max(contours, key=cv2.contourArea)
                
                # Calculate the centroid of the pupil contour
                moments = cv2.moments(pupil_contour)
                if moments['m00'] == 0:
                    continue
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])

                # Calculate the radius of the pupil
                radius = int(cv2.minEnclosingCircle(pupil_contour)[1])

                # Draw the circle around the pupil on the frame
                cv2.circle(frame, (ex+cx, ey+cy), radius, (0, 255, 0), 2)

        # Display the frame with the circle around the pupil
        cv2.imshow('Eye Video', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

# Release the VideoCapture object and destroy all windows
cap.release()
cv2.destroyAllWindows()