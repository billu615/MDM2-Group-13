import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from scipy.cluster.vq import kmeans, vq

def load_image(image_path):
    image = io.imread(image_path)
    M, N, _ = image.shape
    pixels = image.reshape(M * N, 3).astype(float)
    return image, pixels, M, N

def k_means_cluster(pixels, M, N, K):
    centroids, _ = kmeans(pixels, K)
    labels, _ = vq(pixels, centroids)

    colouring = np.random.randint(0, 255, size=(K, 3))
    coloured_image = colouring[labels].reshape(M, N, 3).astype(np.uint8)

    return coloured_image, labels


def main():
    image, pixels, M, N = load_image('ferarri.jpg')

    K = int(input("K: "))

    coloured_image, labels = k_means_cluster(pixels, M, N, K)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('original')

    plt.subplot(1, 2, 2)
    plt.imshow(coloured_image)
    plt.title(f'segmented pic with {K} clusters')

    plt.tight_layout()
    plt.show()


# Run the main function
if __name__ == "__main__":
    main()
