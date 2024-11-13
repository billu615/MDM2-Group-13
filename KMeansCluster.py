import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from sklearn.cluster import KMeans

# Function to load an image and reshape it into pixels
def load_image(image_path):
    image = io.imread(image_path)
    M, N, _ = image.shape
    pixels = image.reshape(M * N, 3).astype(np.float32)  # Ensure float32 type for k-means
    return image, pixels, M, N

# Function to perform K-means with combined spatial and RGB centroids
def combined_kmeans(image, n_clusters=5, alpha=1.0, beta=1.0):
    M, N, _ = image.shape
    x_coords, y_coords = np.meshgrid(np.arange(N), np.arange(M))
    x_coords = x_coords / N
    y_coords = y_coords / M
    rgb_flat = image.reshape(-1, 3) / 255.0  # Normalize RGB values
    x_flat = x_coords.flatten().reshape(-1, 1)
    y_flat = y_coords.flatten().reshape(-1, 1)
    features = np.hstack((alpha * x_flat, alpha * y_flat, beta * rgb_flat))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    labels = kmeans.labels_.reshape(M, N)
    return labels, kmeans.inertia_

# Function to create colored segments based on K-means labels
def create_colored_segments(labels, M, N, K):
    colors = [
        [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0],
        [255, 105, 180], [255, 255, 0], [0, 0, 255]
    ]
    if K > len(colors):
        for _ in range(K - len(colors)):
            colors.append([np.random.randint(0, 256) for _ in range(3)])
    colored_segments = np.zeros((M, N, 3), dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            cluster_id = labels[i, j]
            colored_segments[i, j] = colors[cluster_id]
    return colored_segments

# Function to generate elbow plot
# Function to generate black-and-white elbow plot with solid black points and lines
def plot_elbow_curve(image, image_name, max_clusters=10, alpha=1.0, beta=1.0):
    inertia_values = []
    for K in range(2, max_clusters):
        _, inertia = combined_kmeans(image, n_clusters=K, alpha=alpha, beta=beta)
        inertia_values.append(inertia)

    plt.figure()
    plt.plot(range(2, max_clusters), inertia_values, 'k-')  # Solid black line
    plt.scatter(range(2, max_clusters), inertia_values, color='black')  # Solid black points
    plt.savefig(f"elbow_{image_name}.jpg", dpi=300, format="jpg", bbox_inches="tight")
    plt.close()


# Main function to load images, apply k-means for K from 2 to 10, and save plots
def main():
    # Define the paths to your images
    image_x_path = 'lenna.jpg'  # Replace with actual path to image x
    image_y_path = 'scenery.png'    # Replace with actual path to image y
    image_z_path = "tulips.png"  # Replace with actual path to image z

    # Load all three images
    image_x, _, M_x, N_x = load_image(image_x_path)
    image_y, _, M_y, N_y = load_image(image_y_path)
    image_z, _, M_z, N_z = load_image(image_z_path)

    # Define alpha and beta values (can be adjusted)
    alpha = 0.4  # Adjust as needed
    beta = 2.0  # Adjust as needed

    # Generate elbow plots
    plot_elbow_curve(image_x, "Original", max_clusters=10, alpha=alpha, beta=beta)
    plot_elbow_curve(image_y, "Salt and Pepper", max_clusters=10, alpha=alpha, beta=beta)
    plot_elbow_curve(image_z, "Gaussian", max_clusters=10, alpha=alpha, beta=beta)

    # Loop over values of K from 2 to 10 to generate segmented images
    for K in range(2, 11):
        # Apply K-means to all three images (x, y, z) using combined_kmeans
        labels_x, _ = combined_kmeans(image_x, n_clusters=K, alpha=alpha, beta=beta)
        labels_y, _ = combined_kmeans(image_y, n_clusters=K, alpha=alpha, beta=beta)
        labels_z, _ = combined_kmeans(image_z, n_clusters=K, alpha=alpha, beta=beta)

        # Create colored segments
        colored_segments_x = create_colored_segments(labels_x, M_x, N_x, K)
        colored_segments_y = create_colored_segments(labels_y, M_y, N_y, K)
        colored_segments_z = create_colored_segments(labels_z, M_z, N_z, K)

        # Create a figure with 6 subplots for the original and segmented images
        plt.figure(figsize=(12, 8))

        # Top row: Original images
        plt.subplot(2, 3, 1)
        plt.imshow(image_x)
        plt.title('Original Image (x)')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(image_y)
        plt.title('Salt and Pepper Image (y)')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(image_z)
        plt.title('Gaussian Image (z)')
        plt.axis('off')

        # Bottom row: Colored Segments
        plt.subplot(2, 3, 4)
        plt.imshow(colored_segments_x)
        plt.title(f'Original Segments (K={K})')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(colored_segments_y)
        plt.title(f'Salt and Pepper Segments (K={K})')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.imshow(colored_segments_z)
        plt.title(f'Gaussian Segments (K={K})')
        plt.axis('off')

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(f"nig={K}.jpg")
        plt.close()  # Close the figure to save memory

if __name__ == "__main__":
    main()
