#cv2 is used for image processing, and
#numpy helps with matrix operations and mathematical functions.

import cv2
import numpy as np

# Function to find the focal length
# Uses the known distance to the object, its actual width, and its width in the reference image to calculate the focal length
def find_focal_length(known_distance, known_width, width_in_rf_image):
    return (width_in_rf_image * known_distance) / known_width

# Function to find the distance of an object from the camera
# Takes the focal length, known width of the object, and its width in the current frame to compute how far the object is from the camera.

def find_distance(focal_length, known_width, width_in_frame):
    return (known_width * focal_length) / width_in_frame

# Known distance from camera to object (in cm)
KNOWN_DISTANCE = 50.0  # Adjust this value to your known distance

# Known width of the object (in cm)
KNOWN_WIDTH = 14.0  # Adjust this value to your known object width

# Initialize video capture
# Opens the video capture from the default webcam (0 represents the primary webcam).
cap = cv2.VideoCapture(0)

# Raises an error if the camera fails to open.
if not cap.isOpened():
    raise Exception("Could not open video device")

# Set video width and height (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Capture a frame for calibration
# Grabs a frame from the video feed to use as a reference. If no frame is captured, an error is raised.
ret, ref_image = cap.read()
if not ret:
    raise Exception("Failed to capture reference image. Ensure your camera is working properly.")


# Convert the reference image to grayscale
ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
#Applies Gaussian blur to reduce noise.
# Uses Canny edge detection to detect the edges in the image.
ref_blurred = cv2.GaussianBlur(ref_gray, (5, 5), 0)
ref_edged = cv2.Canny(ref_blurred, 50, 150)

# Find contours in the reference image
ref_contours, _ = cv2.findContours(ref_edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not ref_contours:
    raise ValueError("No contours found in the reference image. Adjust your object or camera settings.")

# Assumes the largest contour is the object and calculates its width by fitting a bounding rectangle around it.
ref_largest_contour = max(ref_contours, key=cv2.contourArea)
_, _, ref_width, _ = cv2.boundingRect(ref_largest_contour)


# Calculate the focal length, calls the user defined function 
focal_length = find_focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_width)
print(f"Focal Length: {focal_length}")

# Function to detect object and calculate size and distance
def detect_object_and_calculate_distance(frame, focal_length, known_width):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assuming the largest contour is the object
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate distance
        distance = find_distance(focal_length, known_width, w)
        
        # Uses the width and height to compute the diagonal size in pixels
        # Calculate the diagonal size of the object in pixels
        diagonal_pixel_size = np.sqrt(w*2 + h*2)
        
        # Calculate the diagonal size of the object in cm
        object_size = (diagonal_pixel_size / ref_width) * known_width
        
        # Draw bounding box and information on frame
        # Displays the object's distance and size in centimeters on the video feed.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Size: {object_size:.2f} cm", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame, distance, object_size
    else:
        return frame, None, None

# Main loop for real-time object detection and distance measurement
# Continuously captures frames from the video feed. If frame capture fails, the loop breaks.
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame. Ensure your camera is working properly.")
        break
    
    # Detect object and calculate distance
    frame, distance, object_size = detect_object_and_calculate_distance(frame, focal_length, KNOWN_WIDTH)
    # Calls the detection function and displays the processed frame with object details.
    # Display the frame
    cv2.imshow('Object Detection and Distance Measurement', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()