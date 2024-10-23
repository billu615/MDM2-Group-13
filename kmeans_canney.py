import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Load the Lenna image
image = cv2.imread('lenna.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Preprocessing for K-means
# Reshape image to a 2D array of pixels (n_pixels x 3)
pixel_values = image_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Step 3: Apply K-means clustering
k = 5  # Number of clusters (adjustable parameter)
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(pixel_values)

# Convert back to the original image shape
segmented_image = labels.reshape(image_rgb.shape[:2])

# Step 4: Edge detection using Canny
# Convert image to grayscale for Canny
gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Apply GaussianBlur to smoothen image before Canny
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)

# Apply Canny edge detection
edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)

# Step 5: Combine K-means segmentation with Canny edges
# Create an empty image to overlay the edges on the segmented image
combined_image = np.zeros_like(image_rgb)

# Iterate over each pixel and combine the segmentation with the edges
for i in range(combined_image.shape[0]):
    for j in range(combined_image.shape[1]):
        # If there's an edge, keep the edge pixel black
        if edges[i, j] > 0:
            combined_image[i, j] = [0, 0, 0]
        else:
            # Otherwise, color the pixel according to its cluster label
            combined_image[i, j] = kmeans.cluster_centers_[labels[i * image_rgb.shape[1] + j]]

# Convert to uint8 for proper visualization
combined_image = np.uint8(combined_image)

# Step 6: Post-process using morphological closing to smooth the result
# Kernel for morphological operations
kernel = np.ones((3, 3), np.uint8)

# Apply closing operation to fill small holes in the segmentation
final_image = cv2.morphologyEx(combined_image, cv2.MORPH_CLOSE, kernel)

# Step 7: Visualize the results
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(image_rgb)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(segmented_image, cmap='gray')
axs[1].set_title('K-Means Segmentation')
axs[1].axis('off')

axs[2].imshow(edges, cmap='gray')
axs[2].set_title('Canny Edge Detection')
axs[2].axis('off')

axs[3].imshow(final_image)
axs[3].set_title('Combined Result')
axs[3].axis('off')

plt.savefig('kmeans_canney.png')
plt.show()
