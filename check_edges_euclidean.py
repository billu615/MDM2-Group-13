import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_threshold(img, sigma=0.33):
    """Function returning the suggested lower and upper thresholds for Canny"""
    # Compute the median of the single channel pixel intensities
    v = np.median(img)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return lower, upper

def salt_pepper(img, salt_prob, pepper_prob):
    """Function returning a noisy version of the image passed in using salt and pepper noise"""
    noisy_image = np.copy(img)
    row,col = img.shape
    num_salt = np.ceil(salt_prob * img.size)
    coords = [np.random.randint(0, i, int(num_salt)) for i in img.shape]
    noisy_image[coords[0], coords[1]] = 255
    
    # Adding pepper noise (black pixels)
    num_pepper = np.ceil(pepper_prob * img.size)
    coords = [np.random.randint(0, i, int(num_pepper)) for i in img.shape]
    noisy_image[coords[0], coords[1]] = 0
    
    return noisy_image


def check_edges(true_edges, noisy_edges):

    n_rows = len(true_edges)
    n_cols = len(true_edges[0])

    true_edge_index = []
    noisy_edge_index = []
    # find edges
    for i in range(n_cols):
        for j in range(n_rows):
            if noisy_edges[i][j] == 255:
                noisy_edge_index.append(np.array([i,j]))
            if true_edges[i][j] == 255:
                true_edge_index.append(np.array([i,j]))
    print(f"{len(noisy_edge_index)} edge pixels")
    # find shortest distances between edges in true and noisy images
    min_distances = []
    for noisy_i in noisy_edge_index:
        min_dist = np.inf
        for true_i in true_edge_index: 
            dist = np.linalg.norm(np.array([noisy_i[0],noisy_i[1]]) - np.array([true_i[0],true_i[1]]))
            if dist < min_dist:
                min_dist = dist
        min_distances.append(min_dist)

    print(np.mean(min_distances))


# read image
image = cv2.imread("./blurred_circle.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("./lena.jpg", cv2.IMREAD_GRAYSCALE)

sigmas = [0.33, 0.33, 0.85, 0.85]

# add noise
seasoned = salt_pepper(image, salt_prob=0.2, pepper_prob=0.2)

# gaussian blur
blurred_img =cv2.GaussianBlur(image, (5,5), 1.4)
blurred_noise =cv2.GaussianBlur(seasoned, (5,5), 1.4)

# perform Canny

# set thresholds
#t1_1,t2_1 = get_threshold(blurred_img, sigmas[0])
#T1_1,T2_1 = get_threshold(blurred_noise, sigmas[0])
#t1_2,t2_2 = get_threshold(blurred_img, sigmas[2])
#T1_2,T2_2 = get_threshold(blurred_noise, sigmas[2])


t1_1, t2_1 = (190, 300)
T1_1, T2_1 = (190, 300)
t1_2, t2_2 = (80, 180)
T1_2, T2_2 = (80, 180)

new1 = cv2.Canny(blurred_img, t1_1,t2_1)
new2 = cv2.Canny(blurred_img, t1_2,t2_2)
new1_noise = cv2.Canny(blurred_noise, T1_1,T2_1)
new2_noise = cv2.Canny(blurred_noise, T1_2,T2_2)

check_edges(new1,new1_noise)
