import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

from scipy.signal import savgol_filter

# Load the grayscale image
image_path = "seasoned_lenna.jpg"  # Change this to your image path
grayscale = io.imread(image_path)

# Check if the image is already in grayscale
if grayscale.ndim == 3:  # If it's an RGB image
    grayscale = color.rgb2gray(grayscale)  # Convert to grayscale

# Set parameters for the Savitzky-Golay filter
window_length = 15  # Must be odd and less than or equal to the size of the image
order = 3  # Polynomial order

# Apply the Savitzky-Golay filter
filtered_image = savgol_filter(grayscale, window_length=window_length, polyorder=order)

# Save the filtered image
output_path = "filtered_image.jpg"  # Change the output path and filename as needed
io.imsave(output_path, (filtered_image * 255).astype(np.uint8))  # Scale to 0-255 and convert to uint8

# Plotting results
plt.figure(figsize=(10, 5))

# Original grayscale image
plt.subplot(1, 2, 1)
plt.imshow(grayscale, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Filtered image
plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image with Savitzky-Golay')
plt.axis('off')

plt.tight_layout()
plt.show()

fig2 = plt.figure(figsize=(6, 6))
plt.imshow(filtered_image, cmap='gray')
plt.axis('off')
plt.savefig('filtered_image.jpg')
plt.show()