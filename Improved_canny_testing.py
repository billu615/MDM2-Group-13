import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_threshold(img, sigma=0.33):
    # Compute the median of the single channel pixel intensities
    v = np.median(img)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return lower, upper

def salt_pepper(img, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    row,col = img.shape
    num_salt = np.ceil(salt_prob * img.size)
    coords = [np.random.randint(0, i, int(num_salt)) for i in img.shape]
    noisy_image[coords[0], coords[1]] = 255
    
    # Adding pepper noise (black pixels)
    num_pepper = np.ceil(pepper_prob * img.size)
    coords = [np.random.randint(0, i, int(num_pepper)) for i in img.shape]
    noisy_image[coords[0], coords[1]] = 0
    
    return noisy_image
def obtain_thresholds(k_,sigma_,E_):
    T_upper = E_ + (k_*sigma_)
    T_lower = T_upper / 2
    return T_lower, T_upper
def show_new(images):

    font_size = 10
    # plot results
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.set_title("Original image", fontsize=font_size)
    ax1.set_axis_off()
    ax1.imshow(images[0], cmap='gray')
    #ax2.set_title(f"t1 = {t1a:.2f}, t2 = {t2a:.2f}")

    ax2.set_axis_off()
    ax2.set_title(f"Noisy image, fontsize=font_size", fontsize=font_size)
    ax2.imshow(images[1], cmap='gray')
    #ax3.set_title(f"t1 = {t1b:.2f}, t2 = {t2b:.2f}")

    ax3.set_axis_off()
    ax3.set_title(f"Newtonian Canny (without noise)", fontsize=font_size)
    ax3.imshow(images[2], cmap='gray')
    #ax4.set_title(f"t1 = {t1c:.2f}, t2 = {t2c:.2f}")

    ax4.set_title(f"Newtonian Canny (with noise)", fontsize=font_size)
    ax4.set_axis_off()
    ax4.imshow(images[3], cmap='gray')
    fig.suptitle("Testing the 'Newtonian' Canny Algorithm with added noise", fontsize=font_size)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)


    fig.tight_layout()

    fig.savefig("Newtonian_Canny_noisy.png")
    plt.show()

def add_gaussian_noise(img, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, img.shape)
    
    noisy_image = img + gaussian_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def show_seven(images):
    
    font_size = 8
    
    fig, ((ax1, ax2, ax3),(ax4, ax6, ax8),(ax5,ax7,ax9)) = plt.subplots(3,3)
    fig.set_size_inches(5,5.5)
    axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
    titles = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)', '(i)']
    for i, ax in enumerate(axes):
        ax.set_axis_off() 
        ax.set_title(titles[i], fontsize=font_size)
        ax.imshow(images[i], cmap='gray')
            
    
    
    fig.subplots_adjust(wspace=0.005, hspace=0.005)
    fig.tight_layout()
    fig.savefig("noise_canny_circle.png")

def show_six(images):

    font_size = 10
    # plot results
    fig = plt.figure(figsize=(5,4))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    ax1.set_title("Original image", fontsize=font_size)
    ax1.set_axis_off()
    ax1.imshow(images[0], cmap='gray')
    #ax2.set_title(f"t1 = {t1a:.2f}, t2 = {t2a:.2f}")

    ax2.set_axis_off()
    ax4.set_title(f"Noisy image", fontsize=font_size)
    ax4.imshow(images[1], cmap='gray')
    #ax3.set_title(f"t1 = {t1b:.2f}, t2 = {t2b:.2f}")

    ax3.set_axis_off()
    ax2.set_title(f"k = {ks[0]} (no noise)", fontsize=font_size)
    ax2.imshow(images[2], cmap='gray')
    #ax4.set_title(f"t1 = {t1c:.2f}, t2 = {t2c:.2f}")

    ax5.set_title(f"k = {ks[1]} (noise)", fontsize=font_size)
    ax4.set_axis_off()
    ax5.imshow(images[3], cmap='gray')

    ax3.set_title(f"k = {ks[2]} (no noise)", fontsize=font_size)
    ax5.set_axis_off()
    ax3.imshow(images[4], cmap='gray')

    ax6.set_title(f"k = {ks[3]} (noise)", fontsize=font_size)
    ax6.set_axis_off()
    ax6.imshow(images[5], cmap='gray')

    fig.suptitle("Testing the 'Newtonian' Canny Algorithm ", fontsize=font_size)
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    fig.tight_layout()

    fig.savefig("Noisy_canny.png")
    plt.show()


def visualise_field(x_max,y_max,v_x,v_y):  
    x, y = np.meshgrid(np.linspace(0, x_max, x_max), np.linspace(0, y_max, y_max))
    
    v_y = v_y[::-1]
    # Plot the vector field
    plt.quiver(x, y, v_x, v_y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Vector Field for Image')
    plt.savefig("vector_field.png")
    plt.show()


#Load an image in grayscale
image = cv2.imread("./lena.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("./cropped_circle.png", cv2.IMREAD_GRAYSCALE)

#image = cv2.imread("./lena.jpg", cv2.IMREAD_GRAYSCALE)
#image = cv2.imread("./Lena_image.jpg", cv2.IMREAD_GRAYSCALE)
salt_noisy = salt_pepper(image, 0.2 ,0.2)
gaus_noisy = add_gaussian_noise(image, sigma=10)
# blur image
blurred_im = cv2.GaussianBlur(image, (5,5), 1.4)

noise_blurred = cv2.GaussianBlur(salt_noisy, (5,5), 1.4)
noise_blurred2 = cv2.GaussianBlur(gaus_noisy, (5,5), 1.4)
new_images = []

# perform traditional canny
new_images.append(cv2.Canny(noise_blurred, 190,300))
new_images.append(cv2.Canny(noise_blurred2, 190,300))


blurred_imgs = [noise_blurred, noise_blurred2]
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
    ks = [0.5, 1.4]
    #t1a,t2a = obtain_thresholds(ks[1],sigma,E_ave)  # if sigma is large k should be small, and vice versa 
    #new_image1 = cv2.Canny(blurred, t1a, t2a)

    #new_image1 = cv2.Canny(blurred,190,300)

    t1b,t2b = obtain_thresholds(ks[n],sigma,E_ave)  # if sigma is large k should be small, and vice versa 
    print(t1b, t2b)
    new_images.append(cv2.Canny(blurred, t1b, t2b))

#t1c,t2c = obtain_thresholds(ks[6],sigma,E_ave)  # if sigma is large k should be small, and vice versa 
#new_image3 = cv2.Canny(blurred, t1c, t2c)


# visualise vector field
#visualise_field(n_cols,n_rows,E_vect[:,:,0], E_vect[:,:,1])

t1,t2 = get_threshold(noise_blurred, 0.33)
T1,T2 = get_threshold(noise_blurred2, 0.33)
new_images.append(cv2.Canny(noise_blurred, t1,t2))
new_images.append(cv2.Canny(noise_blurred, T1,T2))


# plot new image

six_imgs = [image, noise_blurred, new_images[0], new_images[1], new_images[2], new_images[3]]
seven_imgs = [image, noise_blurred, noise_blurred2, new_images[0], new_images[1], new_images[2], new_images[3], new_images[4] ,new_images[5]]
#show_new(images=[image, noisy, new_images[0], new_images[1]])

show_seven(images=seven_imgs)
plt.show()