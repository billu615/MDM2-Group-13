import numpy as np
import matplotlib.pyplot as plt
import cv2


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
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255
    
    # Adding pepper noise (black pixels)
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    
    return noisy_image

# load an image in grayscale
image = cv2.imread("./lena.jpg", cv2.IMREAD_GRAYSCALE)

probs = [0.1, 0.2, 0.3]

sigmas = [0.33, 0.33, 0.85, 0.85]
# add noise
seasoned = salt_pepper(image, salt_prob=probs[1], pepper_prob=probs[1])


# perform canny algorithm
original_blur = cv2.GaussianBlur(image, (5,5), 1.4)
seasoned_blur = cv2.GaussianBlur(seasoned, (5,5), 1.4)

t1_1,t2_1 = get_threshold(original_blur, sigmas[0])
T1_1,T2_1 = get_threshold(seasoned_blur, sigmas[0])
t1_2,t2_2 = get_threshold(original_blur, sigmas[2])
T1_2,T2_2 = get_threshold(seasoned_blur, sigmas[2])

#print(t1, t2, T1, T2)

t1_1, t2_1 = (190, 300)
t1_2, t2_2 = (80, 180)
T1_1, T2_1 = (190, 300)
T1_2, T2_2 = (80, 180)

new1 = cv2.Canny(original_blur, t1_1,t2_1)
new2 = cv2.Canny(original_blur, t1_2,t2_2)
new3 = cv2.Canny(seasoned_blur, T1_1,T2_1)
new4 = cv2.Canny(seasoned_blur, T1_2,T2_2)


#canny_traditional = cv2.Canny(original_blur, 190,255)
#canny_seasoned = cv2.Canny(seasoned_blur, 190,255)

fig,((ax1,ax3,ax5),(ax2,ax4, ax6)) = plt.subplots(2,3)
fig.set_size_inches(6.5,5)
font_size = 10
ax1.set_title("Original image", fontsize=font_size)
ax1.imshow(image, cmap='grey')
ax1.set_axis_off()


ax2.set_title(f"Noisy image", fontsize=font_size)
ax2.imshow(seasoned, cmap='grey')
ax2.set_axis_off()


#ax3.set_title(f"Salt Prob: {probs[1]:.2f}, Pepper Prob: {probs[1]:.2f}")
ax3.set_title(f'$T_l = ${t1_1}, $T_h = ${t2_1} (no noise)', fontsize=font_size)
ax3.imshow(new1, cmap='grey')
ax3.set_axis_off()

#ax4.set_title(f"Salt Prob: {probs[2]:.2f}, Pepper Prob: {probs[2]:.2f}")
ax4.set_title(f'$T_l = ${T1_1}, $T_h = ${T2_1} (noise)', fontsize=font_size)
ax4.imshow(new3, cmap='grey')
ax4.set_axis_off()

ax5.set_title(f'$T_l = ${t1_2}, $T_h = ${t2_2} (no noise)', fontsize=font_size)
ax5.imshow(new2, cmap='grey')
ax5.set_axis_off()

ax6.set_title(f'$T_l = ${T1_2}, $T_h = ${T2_2} (noise)', fontsize=font_size)
ax6.imshow(new4, cmap='grey')
ax6.set_axis_off()

#fig.suptitle("Testing the Traditional Canny Algorithm", fontsize=font_size)
fig.subplots_adjust(wspace=0.1, hspace=0.1)
fig.tight_layout()

fig.savefig("basic_canny.png")
plt.show()