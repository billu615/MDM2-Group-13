# KMeansMethod.py
import numpy as np
from skimage import io, filters, color
from skimage.feature import canny
from scipy.cluster.vq import kmeans, vq


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

    def k_means_cluster(self, K):
        centroids, _ = kmeans(self.pixels, K)
        labels, _ = vq(self.pixels, centroids)
        return centroids, labels

    def create_border_map(self, labels, n):
        border_map = np.zeros((self.M, self.N), dtype=int)
        for i in range(self.M):
            for j in range(self.N):
                current_label = labels[i * self.N + j]
                neighbors = []

                for k in range(-n, n + 1):
                    if k != 0:
                        # Check up and down
                        if 0 <= i + k < self.M:
                            neighbors.append((i + k, j))
                        # Check left and right
                        if 0 <= j + k < self.N:
                            neighbors.append((i, j + k))

                for x, y in neighbors:
                    neighbor_label = labels[x * self.N + y]
                    if neighbor_label != current_label:
                        border_map[i, j] = 1
                        break
        return border_map
