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
