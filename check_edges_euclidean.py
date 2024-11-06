import numpy as np
import matplotlib.pyplot as plt
import cv2

def obtain_thresholds(k_,sigma_,E_):
    T_upper = E_ + (k_*sigma_)
    T_lower = T_upper / 2
    return T_lower, T_upper


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
    """Function to compare segmentations with and without noise"""
    
    # Find edges
    true_edge_index = np.argwhere(true_edges == 255)
    noisy_edge_index = np.argwhere(noisy_edges == 255)

    # Find shortest distances between edges in true and noisy images
    min_distances1 = []
    count = 0
    for noisy_i in noisy_edge_index:
        count+=1
        distances = np.linalg.norm(true_edge_index - noisy_i, axis=1)
        if count == 1:
            print(distances.shape)
        min_distances1.append(np.min(distances))

    # Find shortest distances between every edge in true and noisy images
    min_distances2 = []
    for true_i in true_edge_index:
        distances = np.linalg.norm(noisy_edge_index - true_i, axis=1)
        min_distances2.append(np.min(distances))
    
    mean1 = np.mean(min_distances1)
    mean2 = np.mean(min_distances2)


    return np.mean([mean1,mean2])


def add_gaussian_noise(img, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, img.shape)
    
    noisy_image = img + gaussian_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def traditional(image, salty, gauss):
    new_images = []

    # gaussian blur
    blurred_img =cv2.GaussianBlur(image, (5,5), 1.4)
    blurred_salt =cv2.GaussianBlur(salty, (5,5), 1.4)
    blurred_gauss =cv2.GaussianBlur(gauss, (5,5), 1.4)
    
    # Perform Canny
    new_images.append(cv2.Canny(blurred_img, 80, 180))
    new_images.append(cv2.Canny(blurred_salt, 190, 300))
    new_images.append(cv2.Canny(blurred_gauss, 100, 255))

    return new_images

def median_based(image, salty, gauss, sigma=0.33):
    """median based canny algorithm"""
    new_images = []

    # gaussian blur
    blurred_img =cv2.GaussianBlur(image, (5,5), 1.4)
    blurred_salt =cv2.GaussianBlur(salty, (5,5), 1.4)
    blurred_gauss =cv2.GaussianBlur(gauss, (5,5), 1.4)
    
    # set thresholds
    t1_1,t2_1 = get_threshold(blurred_img, sigma)
    t1_2,t2_2 = get_threshold(blurred_salt, sigma)
    t1_3,t2_3 = get_threshold(blurred_gauss, sigma)

    # perform Canny
    new_images.append(cv2.Canny(blurred_img, t1_1,t2_1))
    new_images.append(cv2.Canny(blurred_salt, t1_2,t2_2))  
    new_images.append(cv2.Canny(blurred_gauss, t1_3,t2_3))

    return new_images

def newtonian(image, salty, gauss):

    # blur image
    blurred_im = cv2.GaussianBlur(image, (5,5), 1.4)

    noise_blurred = cv2.GaussianBlur(salty, (5,5), 1.4)
    noise_blurred2 = cv2.GaussianBlur(gauss, (5,5), 1.4)
    new_images = []


    blurred_imgs = [blurred_im, noise_blurred, noise_blurred2]
    # apply to newton's laws
    G = np.sqrt(1/2)
    for n, blurred in enumerate(blurred_imgs):
            
        n_rows = len(blurred)
        n_cols = len(blurred[0])
        size =  n_rows * n_cols
        E = np.zeros((blurred.shape), np.float64)  # array for gradient magnitudes
        E_vect = np.zeros((n_cols,n_rows,2), np.float64) # array for gradient vectors

        m = blurred  # array for pixel 'masses'
        E_sum = 0
        sum_count = 0
        on_edge = [0,n_rows-1]  # indicies where the pixel is on the edge 

        # search through pixel array and calculate E
        for i in range(n_cols):
            for j in range(n_rows):
                # calculate grad in x and y directions with adjacent pixels
                if i in on_edge or j in on_edge:
                    continue
                else:
                    # gradient in x direction
                    Ex_total = G * ((m[i+1][j]-m[i-1][j])+
                                    (np.sqrt(2)/4)*
                                    (m[i+1][j-1]-m[i-1][j+1]
                                    +m[i+1][j+1]-m[i-1][j-1]))
                    # gradient in y direction 
                    Ey_total = G * ((m[i][j+1]-m[i][j-1])+
                                    (np.sqrt(2)/4)*
                                    (m[i-1][j+1]-m[i+1][j-1]
                                    +m[i+1][j+1]-m[i-1][j-1]))
                    # gradient magnitude
                    E[i][j] = np.sqrt(Ex_total**2 + Ey_total**2)
                    E_vect[i][j][0] = Ex_total
                    E_vect[i][j][1] = Ey_total
                    E_sum += E[i][j]
                    sum_count += 1
        # calculate average E 
        E_ave = E_sum / sum_count

        # calculate standard deviation (sigma)
        sigma = 0
        for i in range(n_cols):
            for j in range(n_rows):
                sigma += ((abs(E[i][j]-E_ave))**2)/sum_count

        sigma = np.sqrt(sigma)

        # determine optimal threshold values
        ks = [0.6, 1.4, 0.8]
        #t1a,t2a = obtain_thresholds(ks[1],sigma,E_ave)  # if sigma is large k should be small, and vice versa 
        #new_image1 = cv2.Canny(blurred, t1a, t2a)

        #new_image1 = cv2.Canny(blurred,190,300)

        t1b,t2b = obtain_thresholds(ks[n],sigma,E_ave)  # if sigma is large k should be small, and vice versa 
        print(t1b, t2b)
        new_images.append(cv2.Canny(blurred, t1b, t2b))
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(new_images[1])
    ax2.imshow(new_images[0])
    plt.show()
    return new_images


# read image
circle = cv2.imread("./blurred_circle.jpg", cv2.IMREAD_GRAYSCALE)
circle_gaus = cv2.imread("./Images/n_gaus_circle.jpg", cv2.IMREAD_GRAYSCALE)
circle_salty = cv2.imread("./Images/n_gaus_circle.jpg", cv2.IMREAD_GRAYSCALE)

lenna = cv2.imread("./lena.jpg", cv2.IMREAD_GRAYSCALE)
lenna_gaus = cv2.imread("./Images/n_gaus_lenna.jpg", cv2.IMREAD_GRAYSCALE)
lenna_salty = cv2.imread("./Images/n_snp_lenna.jpg", cv2.IMREAD_GRAYSCALE)
# get result for traditional method
traditional_imgs = traditional(lenna, lenna_salty, lenna_gaus)

# get result for Newtonian method
newton_imgs = newtonian(lenna, lenna_salty, lenna_gaus)

# get result for median method
median_imgs = median_based(lenna, lenna_salty, lenna_gaus)


print("Min distance value for traditional Canny with salt and pepper:", check_edges(traditional_imgs[0], traditional_imgs[1]))
print("Min distance value for traditional Canny with gaussian: ", check_edges(traditional_imgs[0], traditional_imgs[2]))

print("Min distance value for Newtonian Canny with salt and pepper:", check_edges(newton_imgs[0], newton_imgs[1]))
print("Min distance value for Newtonian Canny with gaussian: ", check_edges(newton_imgs[0], newton_imgs[2]))

print("Min distance value for median-based Canny with salt and pepper:", check_edges(median_imgs[0], median_imgs[1]))
print("Min distance value for median-based Canny with gaussian: ", check_edges(median_imgs[0], median_imgs[2]))

# add noise
#seasoned = salt_pepper(image, salt_prob=0.2, pepper_prob=0.2)

# gaussian blur
#blurred_img =cv2.GaussianBlur(image, (5,5), 1.4)
#blurred_noise =cv2.GaussianBlur(seasoned, (5,5), 1.4)

# perform Canny

# set thresholds
#t1_1,t2_1 = get_threshold(blurred_img, sigmas[0])
#T1_1,T2_1 = get_threshold(blurred_noise, sigmas[0])
#t1_2,t2_2 = get_threshold(blurred_img, sigmas[2])
#T1_2,T2_2 = get_threshold(blurred_noise, sigmas[2])


"""t1_1, t2_1 = (190, 300)
T1_1, T2_1 = (190, 300)
t1_2, t2_2 = (80, 180)
T1_2, T2_2 = (80, 180)

new1 = cv2.Canny(blurred_img, t1_1,t2_1)
new2 = cv2.Canny(blurred_img, t1_2,t2_2)
new1_noise = cv2.Canny(blurred_noise, T1_1,T2_1)
new2_noise = cv2.Canny(blurred_noise, T1_2,T2_2)

check_edges(new1,new1_noise)
"""