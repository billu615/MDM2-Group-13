import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_laplace

def laplacian_of_gaussian(image_path, sigma=2.0, threshold=0.03):
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image {image_path} not found.")

    
    log_image = gaussian_laplace(image, sigma=sigma)

    
    log_image_normalized = np.abs(log_image)
    log_image_normalized = (log_image_normalized - np.min(log_image_normalized)) / (np.max(log_image_normalized) - np.min(log_image_normalized))

    
    edges = log_image_normalized > threshold

    
    edges = edges.astype(np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

    
    segmented_image = image.copy()
    segmented_image[edges == 1] = 255

    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(log_image_normalized, cmap='gray')
    plt.title('Laplacian of Gaussian')

    plt.subplot(1, 3, 3)
    plt.imshow(segmented_image, cmap='gray')
    plt.title('Segmented Image')

    plt.tight_layout()
    plt.show()

    return segmented_image, edges


image_path = 'lena_image.jpg'
segmented_image, edges = laplacian_of_gaussian(image_path, sigma=2.0, threshold=0.03)
