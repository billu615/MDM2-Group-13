import numpy as np
import matplotlib.pyplot as plt
import cv2


image_path = 'lenna.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found. Please check the path.")

f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)
magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)

rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols), np.uint8)
radius = 50
cv2.circle(mask, (ccol, crow), radius, 1, -1)

filtered_shifted = f_transform_shifted * mask
filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_shifted))
filtered_image = np.abs(filtered_image)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')


plt.subplot(1, 3, 2)
plt.title("Magnitude Spectrum")
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')


plt.subplot(1, 3, 3)
plt.title("Filtered Image (Low-Pass)")
plt.imshow(filtered_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('comparison_image.png', format='png')
plt.show()

plt.imshow(filtered_image, cmap='gray')
plt.axis('off')
plt.show()
cv2.imwrite('low_pass_image.png', filtered_image)


