import cv2
import numpy as np
import matplotlib.pyplot as plt

def obtain_thresholds(k_,sigma_,E_):
    T_upper = E_ + (k_*sigma_)
    T_lower = T_upper / 2
    return T_lower, T_upper


#Load an image in grayscale
image2 = cv2.imread("./curious-bird-1-1374322.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("./Lena_image.jpg", cv2.IMREAD_GRAYSCALE)

# blur image
blurred = cv2.GaussianBlur(image, (5,5), 1.4)

# apply to newton's laws
G = np.sqrt(1/2)
#G = 0.5
E = np.zeros((blurred.shape), np.float64)  # array for gradient values
m = blurred  # array for pixel 'masses'
n_rows = len(blurred)
n_cols = len(blurred[0])
size =  n_rows * n_cols
E_sum = 0
sum_count = 0
on_edge = [0,224]  # indicies where the pixel is on the edge 

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
t1a,t2a = obtain_thresholds(1.2,sigma,E_ave)  # if sigma is large k should be small, and vice versa 
new_image1 = cv2.Canny(blurred, t1a, t2a)

t1b,t2b = obtain_thresholds(1.4,sigma,E_ave)  # if sigma is large k should be small, and vice versa 
new_image2 = cv2.Canny(blurred, t1b, t2b)

t1c,t2c = obtain_thresholds(1.6,sigma,E_ave)  # if sigma is large k should be small, and vice versa 
new_image3 = cv2.Canny(blurred, t1c, t2c)



# plot results
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.imshow(image, cmap='gray')
ax2.set_title(f"t1 = {t1a:.2f}, t2 = {t2a:.2f}")
ax2.imshow(new_image1, cmap='gray')
ax3.set_title(f"t1 = {t1b:.2f}, t2 = {t2b:.2f}")
ax3.imshow(new_image2, cmap='gray')
ax4.set_title(f"t1 = {t1c:.2f}, t2 = {t2c:.2f}")
ax4.imshow(new_image3, cmap='gray')
fig.tight_layout()
plt.show()