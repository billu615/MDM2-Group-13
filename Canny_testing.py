import cv2
import numpy as np
import matplotlib.pyplot as plt

#Load an image in grayscale
image2 = cv2.imread("./curious-bird-1-1374322.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("./Lena_image.jpg", cv2.IMREAD_GRAYSCALE)

# blur image
blurred = cv2.GaussianBlur(image, (5,5), 1.4)
# set threshholds
T_lower = [30, 80, 100]
T_upper = [90, 120, 150]
new_images = []
# run Canny algorithm
#new_image1 = cv2.Canny(image, T_lower, T_upper)

for t_high in T_upper:
    for t_low in T_lower:
        new_images.append(cv2.Canny(blurred, t_low, t_high))

thresh1 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, 199, 5) 
  
thresh2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 199, 5) 
# show new image
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.imshow(thresh2, cmap='gray')
ax2.imshow(new_images[0], cmap='gray')
ax3.imshow(new_images[1], cmap='gray')
ax4.imshow(new_images[2], cmap='gray')

plt.show()