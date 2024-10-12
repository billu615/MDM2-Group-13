import cv2
import numpy as np

# Load the image
image = cv2.imread("./Lena_image.jpg", cv2.IMREAD_GRAYSCALE)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 1.4)

def canny_edge_detection(lower_threshold, upper_threshold):
    # Apply Canny edge detection with the current thresholds
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    return edges

# Function to update the thresholds dynamically using trackbars
def update(val):
    lower = cv2.getTrackbarPos('Lower', 'Canny Edge Detection')
    upper = cv2.getTrackbarPos('Upper', 'Canny Edge Detection')
    
    # Call the canny_edge_detection function
    edges = canny_edge_detection(lower, upper)
    
    # Display the result
    cv2.imshow('Canny Edge Detection', edges)

# Create a window
cv2.namedWindow('Canny Edge Detection')

# Create trackbars for Lower and Upper threshold
cv2.createTrackbar('Lower', 'Canny Edge Detection', 50, 255, update)
cv2.createTrackbar('Upper', 'Canny Edge Detection', 150, 255, update)

# Call update once to initialize the display
update(0)

# Wait for the user to press a key, then exit
cv2.waitKey(0)
cv2.destroyAllWindows()
