from PIL import Image
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def select_image(name):
    image = Image.open('initial/'+name+'_gray.jpg').convert('L')
    image_og = np.array(image)
    image = Image.open('initial/n_gaus_'+name+'.jpg').convert('L')
    image_gaus = np.array(image)
    image_noise = Image.open('initial/n_snp_'+name+'.jpg').convert('L')
    image_snp = np.array(image_noise)
    return image_og, image_gaus, image_snp

# Apply Savitzky-Golay filter on both axes
def sg(image_array, window_size, poly_order):
    smoothed_image_array = savgol_filter(image_array, window_size, poly_order, axis=0)
    smoothed_image_array = savgol_filter(smoothed_image_array, window_size, poly_order, axis=1)
    smoothed_image = np.clip(smoothed_image_array, 0, 255).astype('uint8')
    return smoothed_image

# SSIM value
def ssim_compare(img1, img2):
    ssim_score = ssim(img1, img2, full=True)
    return ssim_score

# optimal ssim value and corresponding parameter
def optimal(ssim, parameter):
    max_value = max(ssim)
    max_window = parameter[np.argmax(ssim)]
    return max_value, max_window

image_og, image_gaus, image_snp = select_image('circle')
window_lengths = np.arange(3, 50)

# SNP
print('SNP')
ssim_snp = ssim_compare(image_og, image_snp)[0]
print('SSIM value original vs noised: ', ssim_snp)
ssim_snp_filter = [ssim_compare(image_og, sg(image_snp, i+1, 2))[0] for i in window_lengths]
snp_ssim, snp_window = optimal(ssim_snp_filter, window_lengths)
print('SSIM Value original vs filtered:', snp_ssim)
print('Optimal Window Length: ', snp_window)

# Gaussian
print('Gaussian')
ssim_gaus = ssim_compare(image_og, image_gaus)[0]
print('SSIM value original vs noised: ', ssim_gaus)
ssim_gaus_filter = [ssim_compare(image_og, sg(image_gaus, i+1, 2))[0] for i in window_lengths]
gaus_ssim, gaus_window = optimal(ssim_gaus_filter, window_lengths)
print('SSIM Value original vs filtered:', gaus_ssim)
print('Optimal Window Length: ', gaus_window)

'''
plt.figure()
plt.plot(window_lengths, ssim_scores)
plt.show()


plt.figure()

plt.subplot(1, 2, 1)
plt.imshow(image_og, cmap='gray')
plt.title('original')

plt.subplot(1, 2, 2)
plt.imshow(smoothed_image, cmap='gray')
plt.title('blurred')

plt.show()

# Display the original and smoothed images

plt.figure()
plt.imshow(smoothed_image, cmap='gray')
plt.axis("off")
plt.imsave('gaus_sg_lenna.jpg', smoothed_image, cmap='gray')
'''
