import cv2
import numpy as np

# Read images
canny_original = cv2.imread('gray_lenna.jpg', 0)
canny_noisy = cv2.imread('seasoned_lenna.jpg', 0)


# Function to calculate PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Return infinity if images are identical
    max_pixel_value = 255.0  # For 8-bit images
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


# Calculate and print PSNR
psnr_value = calculate_psnr(canny_original, canny_noisy)
print("PSNR:", psnr_value)
