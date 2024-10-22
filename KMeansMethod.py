import numpy as np
from skimage import io, filters, color
from skimage.feature import canny
from scipy.cluster.vq import kmeans, vq
from scipy.signal import savgol_filter

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image, self.pixels, self.M, self.N = self.load_image()

    def load_image(self):
        image = io.imread(self.image_path)
        M, N, _ = image.shape
        pixels = image.reshape(M * N, 3).astype(float)
        return image, pixels, M, N

    def apply_gaussian_blur(self, sigma):
        grayscale_image = color.rgb2gray(self.image)
        blurred_image = filters.gaussian(grayscale_image, sigma=sigma)
        return blurred_image

    def apply_canny(self, blurred_image):
        edges = canny(blurred_image)
        return edges

    def savitzky_golay(self, window_length, order, derivative=1):
        grayscale_image = color.rgb2gray(self.image)
        filtered_image = savgol_filter(grayscale_image, window_length, order, deriv=derivative, axis=0)
        return filtered_image

    def k_means_cluster(self, K):
        centroids, _ = kmeans(self.pixels, K)
        labels, _ = vq(self.pixels, centroids)
        return centroids, labels

    def create_border_map(self, labels):
        border_map = np.zeros((self.M, self.N), dtype=np.uint8)

        for i in range(self.M):
            for j in range(self.N):
                current_label = labels[i * self.N + j]
                # Check neighbors for different labels
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if (0 <= i + di < self.M) and (0 <= j + dj < self.N):
                            neighbor_label = labels[(i + di) * self.N + (j + dj)]
                            if current_label != neighbor_label:
                                border_map[i, j] = 1  # Mark as border
                                break
                    if border_map[i, j] == 1:
                        break  # No need to check other neighbors

        return border_map
