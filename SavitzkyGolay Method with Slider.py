import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, filters, measure
from scipy.signal import savgol_filter
from matplotlib.widgets import Slider

# Load and process the image
image_path = "stones.jpg"  # Change this to your image path
image = io.imread(image_path)
grayscale = color.rgb2gray(image)

# Fix window length at 21
fixed_window_length = 11

# Savitzky-Golay filter function
def savitzky_golay(image, window_length, order, derivative=1):
    filtered_image = savgol_filter(image, window_length, order, deriv=derivative, axis=0)
    return filtered_image

# Function to update the grayscale segmentation
def update(val):
    order = int(order_slider.val)

    # Recompute the gradients
    grad_x = savitzky_golay(grayscale, window_length=fixed_window_length, order=order, derivative=2)
    grad_y = savitzky_golay(grayscale.T, window_length=fixed_window_length, order=order, derivative=2).T
    grad_mag = np.hypot(grad_x, grad_y)

    # Segment the image
    threshold = filters.threshold_otsu(grad_mag)
    binary_image = grad_mag > threshold
    labelled_image, num_labels = measure.label(binary_image, connectivity=2, return_num=True)

    # Update the grayscale images
    ax_grad_mag.imshow(grad_mag, cmap='gray')
    ax_binary.imshow(binary_image, cmap='gray')
    fig.canvas.draw_idle()

# Create a figure for displaying the grayscale images
fig, (ax_grad_mag, ax_binary) = plt.subplots(2, 1, figsize=(6, 12))

# Initial computation for the gradient magnitude and binary image
grad_x = savitzky_golay(grayscale, window_length=fixed_window_length, order=3, derivative=2)
grad_y = savitzky_golay(grayscale.T, window_length=fixed_window_length, order=3, derivative=2).T
grad_mag = np.hypot(grad_x, grad_y)

threshold = filters.threshold_otsu(grad_mag)
binary_image = grad_mag > threshold
labelled_image, num_labels = measure.label(binary_image, connectivity=2, return_num=True)

# Display the initial gradient magnitude and binary images
ax_grad_mag.imshow(grad_mag, cmap='gray')
ax_grad_mag.set_title('Gradient Magnitude')
ax_binary.imshow(binary_image, cmap='gray')
ax_binary.set_title('Binary Segmented Image')
plt.axis("off")

# Create order slider
ax_order = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
order_slider = Slider(ax_order, 'Order', 2, 7, valinit=3, valstep=1)

# Call update function when slider changes
order_slider.on_changed(update)

plt.tight_layout()
plt.show()
