import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    
    noisy_image = image + gaussian_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

image_path = 'lena_image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

mean = 0
sigma = 10
noisy_1 = add_gaussian_noise(image, mean, sigma)
noisy_2 = add_gaussian_noise(image, mean, 20)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Sigma = 10")
plt.imshow(noisy_1, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Signma = 20")
plt.imshow(noisy_2, cmap='gray')
plt.axis("off")

plt.show()
