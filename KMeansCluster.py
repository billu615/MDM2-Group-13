import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from scipy.cluster.vq import kmeans, vq

def load_image(image_path):
    image = io.imread(image_path)
    M, N, _ = image.shape
    pixels = image.reshape(M * N, 3).astype(float)
    return image, pixels, M, N

def k_means_cluster(pixels, M, N, K):
    centroids, _ = kmeans(pixels, K)
    labels, _ = vq(pixels, centroids)

    colouring = np.random.randint(0,255, size=(K, 3))
    coloured_image = colouring[labels].reshape(M, N, 3).astype(np.uint8)

    return coloured_image, labels



#this part is for wcss calculation, finds distance between data points and centroids
def compute_wcss(pixels, centroids, labels):
    wcss = 0
    for i in range(len(centroids)):
        cluster_points = pixels[labels == i]
        wcss += np.sum((cluster_points - centroids[i]) ** 2)
    return wcss


def elbow_method(pixels, max_k=10):
    wcss_values = []
    K_values = range(1, max_k)
    for K in K_values:
        centroids, _ = kmeans(pixels, K)
        labels, _ = vq(pixels, centroids)
        wcss = compute_wcss(pixels, centroids, labels)
        wcss_values.append(wcss)

    plt.figure(figsize=(8, 5))
    plt.plot(K_values, wcss_values, marker='o', linestyle='--')
    plt.title('Elbow Method: WCSS vs K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')
    plt.show()

    return wcss_values


def main():
    image, pixels, M, N = load_image('Lena_image.jpg')
    #elbow_method(pixels, max_k=10)
    optimal_K = int(input("K: "))
    colored_image, labels = k_means_cluster(pixels, M, N, optimal_K)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(color.rgb2hsv(colored_image))
    plt.title(f'Segmented Image with {optimal_K} clusters (Colored Regions)')

    plt.tight_layout()
    plt.show()


# Run the main function
if __name__ == "__main__":
    main()