import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import savgol_filter
import seaborn as sns
import pandas as pd
from scipy.ndimage import laplace
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
        
        min_distances1.append(np.min(distances))

    # Find shortest distances between every edge in true and noisy images
    min_distances2 = []
    for true_i in true_edge_index:
        distances = np.linalg.norm(noisy_edge_index - true_i, axis=1)
        min_distances2.append(np.min(distances))
        if count == 1:
                print(distances.shape)
    mean1 = np.mean(min_distances1)
    mean2 = np.mean(min_distances2)


    return np.mean([mean1,mean2])


def add_gaussian_noise(img, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, img.shape)
    
    noisy_image = img + gaussian_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def traditional_for_sg(image, salty, gauss):
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
        salty = salty.astype(np.uint8)
        gauss = gauss.astype(np.uint8)

    new_images = []    

    # gaussian blur
    blurred_img =cv2.GaussianBlur(image, (5,5), 1.4)
    blurred_salt =cv2.GaussianBlur(salty, (5,5), 1.4)
    blurred_gauss =cv2.GaussianBlur(gauss, (5,5), 1.4)
    
    # Perform Canny
    new_images.append(cv2.Canny(blurred_img, 25, 60))
    new_images.append(cv2.Canny(blurred_salt, 100, 110))
    new_images.append(cv2.Canny(blurred_gauss, 60, 70))

    return new_images

#traditional canny algorithm
def traditional(image, salty, gauss):
    new_images = []
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
        salty = salty.astype(np.uint8)
        gauss = gauss.astype(np.uint8)

    # gaussian blur
    blurred_img =cv2.GaussianBlur(image, (5,5), 1.4)
    blurred_salt =cv2.GaussianBlur(salty, (5,5), 1.4)
    blurred_gauss =cv2.GaussianBlur(gauss, (5,5), 1.4)
    
    # Perform Canny
    new_images.append(cv2.Canny(blurred_img, 50, 180))
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


def fourier(image, salty, gauss):
    new_images = [] 


    return newton_imgs



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
        ks = [0.1, 1.4, 0.8]

        t1b,t2b = obtain_thresholds(ks[n],sigma,E_ave)  # if sigma is large k should be small, and vice versa 
        print(t1b, t2b)
        new_images.append(cv2.Canny(blurred, t1b, t2b))
    
    return new_images

def laplacian(image, salty, gauss):
    new_images = []
    # gaussian blur
    blurred_img =cv2.GaussianBlur(image, (5,5), 1.4)
    blurred_salt =cv2.GaussianBlur(salty, (5,5), 1.4)
    blurred_gauss =cv2.GaussianBlur(gauss, (5,5), 1.4)

    new_images.append(laplace(blurred_img))
    new_images.append(laplace(blurred_salt))
    new_images.append(laplace(blurred_gauss))

    return new_images

def display_bars(data, x_labels):
    """function to better visualise the data obtained"""
    snp_data = data[:,0]
    gaus_data = data[:,1]

    max_value = max(max(snp_data), max(gaus_data))


    bar_labels = ['1','2','3','4','5']
    size_font = 8
    colours = ["#fc0303", "fc9003", "fc9003", "20913a", "0dadd1"]
    plt.figure()
    plt.subplot(1,2,1)
    for i in range(len(snp_data)):
        plt.bar(bar_labels[i], snp_data[i])
    plt.title("Salt and Pepper Noise", fontsize=size_font)
    plt.ylim(0,max_value)

    #plt.xticks(rotation=45, ha='right')
    plt.xlabel("Method")
    plt.ylabel('Mean Distance Between Edges (pixels)')


    plt.subplot(1,2,2)
    for i in range(len(gaus_data)):
        plt.bar(bar_labels[i], gaus_data[i])
    plt.title("Gaussian Noise", fontsize=size_font)
    #plt.xticks(rotation=45, ha='right')
    plt.ylim(0,max_value)

    plt.xlabel("Method")
    plt.ylabel('Mean Distance Between Edges (pixels)')
    plt.legend(["Traditional Canny", "Newtonian Canny", "Median-Based Canny", "Canny with SG filter"])
    plt.tight_layout()
    plt.show()

    #plt.savefig("distances.png")


# read image
circle = cv2.imread("./blurred_circle.jpg", cv2.IMREAD_GRAYSCALE)
circle_gaus = cv2.imread("./Images/n_gaus_circle.jpg", cv2.IMREAD_GRAYSCALE)
circle_salty = cv2.imread("./Images/n_gaus_circle.jpg", cv2.IMREAD_GRAYSCALE)

lenna = cv2.imread("./lena.jpg", cv2.IMREAD_GRAYSCALE)
lenna_gaus = cv2.imread("./Images/n_gaus_lenna.jpg", cv2.IMREAD_GRAYSCALE)
lenna_salty = cv2.imread("./Images/n_snp_lenna.jpg", cv2.IMREAD_GRAYSCALE)
lennas = [lenna, lenna_salty, lenna_gaus]

# get result for traditional method
traditional_imgs = traditional(lenna, lenna_salty, lenna_gaus)

# get result for Newtonian method
newton_imgs = newtonian(lenna, lenna_salty, lenna_gaus)

# get result for median method
median_imgs = median_based(lenna, lenna_salty, lenna_gaus)

# get result for sg+Canny
sg_imgs = [savgol_filter(img, 25, 3) for img in lennas]
names = ['sg_lenna.jpg', 'sg_snp_lenna.jpg', 'sg_gaus_lenna.jpg']
for i,sg in enumerate(sg_imgs):
    plt.figure()
    plt.imshow(sg, cmap='gray')
    plt.imsave(names[i],sg)
    plt.close()
sg_can_imgs = traditional_for_sg(sg_imgs[0], sg_imgs[1], sg_imgs[2])

# get result for laplacian
laplace_imgs = laplacian(lenna, lenna_salty, lenna_gaus)

trad_snp_dist = check_edges(traditional_imgs[0], traditional_imgs[1])
trad_gauss_dist = check_edges(traditional_imgs[0], traditional_imgs[2])

newt_snp_dist = check_edges(newton_imgs[0], newton_imgs[1])
newt_gauss_dist = check_edges(newton_imgs[0], newton_imgs[2])

med_snp_dist = check_edges(median_imgs[0], median_imgs[1])
med_gauss_dist = check_edges(median_imgs[0], median_imgs[2])

sg_snp_dist = check_edges(sg_can_imgs[0], sg_can_imgs[1])
sg_gauss_dist = check_edges(sg_can_imgs[0], sg_can_imgs[2])

laplace_snp_dist = check_edges(laplace_imgs[0], laplace_imgs[1])
laplace_gaus_dist = check_edges(laplace_imgs[0], laplace_imgs[2])


print("Min distance value for traditional Canny with salt and pepper:", trad_snp_dist)
print("Min distance value for traditional Canny with gaussian: ", trad_gauss_dist)

print("Min distance value for Newtonian Canny with salt and pepper:", newt_snp_dist)
print("Min distance value for Newtonian Canny with gaussian: ", newt_gauss_dist)

print("Min distance value for median-based Canny with salt and pepper:", med_snp_dist)
print("Min distance value for median-based Canny with gaussian: ", med_gauss_dist)

print("Min distance value for SG with traditional Canny with salt and pepper:", sg_snp_dist)
print("Min distance value for SG with traditional Canny with gaussian: ", sg_gauss_dist)

print("Min distance value for the Laplacian filter with salt and pepper:", laplace_snp_dist)
print("Min distance value for the Laplacian filter with gaussian: ", laplace_gaus_dist)

min_distances = np.array([[trad_snp_dist, trad_gauss_dist],
                 [newt_snp_dist, newt_gauss_dist],
                 [med_snp_dist, med_gauss_dist],
                 [sg_snp_dist, sg_gauss_dist],
                 [laplace_snp_dist, laplace_gaus_dist]
                 ])
bar_labels = ["Traditional Canny", "Newtonian Canny", "Median-Based Canny", "SG with Traditional Canny"]

display_bars(min_distances, bar_labels)
