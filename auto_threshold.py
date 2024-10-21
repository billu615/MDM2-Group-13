import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_threshold(img, sigma):
    # Compute the median of the single channel pixel intensities
    v = np.median(img)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return lower, upper

def plot_img(imgs):
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    axs = [ax1, ax2, ax3, ax4]
    axs[1].set_title(f"T_low = {t_0s[0]:.2f}, T_high = {t_1s[0]:.2f}")
    axs[3].set_title(f"T_low = {t_0s[1]:.2f}, T_high = {t_1s[1]:.2f}")

    for i, img in enumerate(imgs):
        axs[i].imshow(img, cmap='grey')   
    fig.tight_layout()    
    plt.show()

#Load an image in grayscale
image = cv2.imread("./Lena_image.jpg", cv2.IMREAD_GRAYSCALE)
#image2 = cv2.imread("./fracture.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("./ferarri.jpg", cv2.IMREAD_GRAYSCALE)

images = [image, image2]
# blur image
blurred = cv2.GaussianBlur(image, (5,5), 1.4)
blurred2 = cv2.GaussianBlur(image2, (5,5), 1.4)
blurs = [blurred, blurred2]
# find thresholds
st_devs = [0.33, 0.5, 1.0, 1.5]
t_0s = np.array([0,0])
t_1s = np.array([0,0])

for i in range(2):
    t_0s[i], t_1s[i] = get_threshold(blurs[i],st_devs[0])

edges = cv2.Canny(blurred, t_0s[0], t_1s[0])
other1 = cv2.Canny(blurred2, t_0s[1], t_0s[1])
# plot different images
plot_img([image, edges, image2, other1])

# vary standard deviation
std_ts = []
for sd in st_devs:
    std_ts.append(get_threshold(blurred2, sd))

#print(t_0s[1],t_1s[1])
#print(std_ts)    
other1 = cv2.Canny(blurred2, std_ts[0][0], std_ts[0][1])
other2 = cv2.Canny(blurred2, std_ts[1][0], std_ts[1][1])
other3 = cv2.Canny(blurred2, std_ts[2][0], std_ts[2][1])
other4 = cv2.Canny(blurred2, std_ts[3][0], std_ts[3][1])


# show varied standard deviation
plot_img([other1, other2, other3, other4])
