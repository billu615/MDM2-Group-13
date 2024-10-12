import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, filters, exposure, measure
from scipy.signal import savgol_filter

image_path = "dog.jpg"
image = io.imread(image_path)
grayscale = color.rgb2gray(image)

window_length= 25 #can change these depending on size of image, 1% of row pixels
order= 6

def savitzky_golay(image, window_length, order, derivative=1):
    filtered_image = savgol_filter(image, window_length, order, deriv=derivative, axis=0) #used package here from scipy.signal
    return filtered_image

grad_x = savitzky_golay(grayscale, window_length=window_length, order=order, derivative=2) #finds gradient in x direction
grad_y = savitzky_golay(grayscale.T, window_length=window_length, order=order, derivative=2).T #finds gradient in y direction

grad_mag = np.hypot(grad_x,grad_y) #magnitude calculation
rescale = exposure.rescale_intensity(grad_mag,in_range=(0,1)) #rescales intensities from what would be 0-255

threshold = filters.threshold_otsu(grad_mag) #read about this being useful for thresholding
binary_image = grad_mag > threshold

labelled_image, num_labels = measure.label(binary_image, connectivity=2, return_num=True)

coloured_segments = np.zeros((*labelled_image.shape,3), dtype= np.uint8)

for label in range(1, num_labels + 1):
    mask = labelled_image == label
    coloured_segments[mask] = np.random.randint(0,255,size=(3,))

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(grad_mag, cmap='gray')
plt.title('original')

plt.subplot(2, 2, 2)
plt.imshow(grad_mag, cmap='gray')
plt.title('gradient magnitude')


plt.subplot(2, 2, 3)
plt.imshow(binary_image, cmap='gray')
plt.title('binary segmented image')


plt.subplot(2, 2, 4)
plt.imshow(coloured_segments)
plt.title('coloured segmented image')


plt.tight_layout()
plt.show()