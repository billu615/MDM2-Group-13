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
    num_salt = np.ceil(salt_prob * image.size)
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
# add noise
seasoned = salt_pepper(image, salt_prob=probs[1], pepper_prob=probs[1])


# perform canny algorithm
original_blur = cv2.GaussianBlur(image, (5,5), 1.4)
seasoned_blur = cv2.GaussianBlur(seasoned, (5,5), 1.4)

t1,t2 = get_threshold(original_blur, 0.33)
T1,T2 = get_threshold(seasoned_blur, 0.33)

canny_bland = cv2.Canny(original_blur, t1,t2)
canny_seasoned = cv2.Canny(seasoned_blur, T1,T2)

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
ax1.set_title("Original image")
ax1.imshow(image, cmap='grey')

ax2.set_title(f"Salt Prob: {probs[1]:.2f}, Pepper Prob: {probs[1]:.2f}")
ax2.imshow(seasoned, cmap='grey')

#ax3.set_title(f"Salt Prob: {probs[1]:.2f}, Pepper Prob: {probs[1]:.2f}")
ax3.set_title('Canny with original image')
ax3.imshow(canny_bland, cmap='grey')

#ax4.set_title(f"Salt Prob: {probs[2]:.2f}, Pepper Prob: {probs[2]:.2f}")
ax4.set_title('Canny with noisy image')
ax4.imshow(canny_seasoned, cmap='grey')
fig.suptitle("Adding Salt and Pepper Noise to the Canny Algorithm ")
fig.tight_layout()
plt.show()