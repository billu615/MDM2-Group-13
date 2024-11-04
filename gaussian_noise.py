import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, sigma=50):
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    
    noisy_image = image + gaussian_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

image_path = 'blurred_circle.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

mean = 0
noisy = add_gaussian_noise(image, mean)

plt.figure(figsize=(10, 5))
plt.imshow(noisy, cmap='gray')
plt.axis("off")
plt.imsave('n_gaus_circle.jpg', noisy, cmap='gray')
