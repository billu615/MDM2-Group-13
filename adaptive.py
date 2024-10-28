import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('lena_image.jpg', cv2.IMREAD_GRAYSCALE)

gaussian_filtered_image = cv2.GaussianBlur(image, (9, 9), sigmaX=10)

bilateral_filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Gaussian Filtered Image")
plt.imshow(gaussian_filtered_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Bilateral Filtered Image")
plt.imshow(bilateral_filtered_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
