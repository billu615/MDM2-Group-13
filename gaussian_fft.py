import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

def gaussian_filter_2d(shape, sigma):
    M, N = shape
    u = np.arange(-M//2, M//2) 
    v = np.arange(-N//2, N//2)
    U, V = np.meshgrid(u, v)
    D = U**2 + V**2
    return np.exp(-D / (2 * sigma**2))

image = plt.imread('lena_image.jpg') 
image_gray = np.mean(image, axis=2)


F_image = fft2(image_gray)
F_image_shifted = fftshift(F_image)


sigma = 50 
gaussian_kernel = gaussian_filter_2d(image_gray.shape, sigma) 

filtered_image_freq = F_image_shifted * gaussian_kernel


filtered_image = np.real(ifft2(fftshift(filtered_image_freq)))


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_gray, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Filtered Image (Gaussian)')
plt.imshow(filtered_image, cmap='gray')
plt.show()
