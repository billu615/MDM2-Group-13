from scipy.ndimage import gaussian_filter
from PIL import Image
import numpy as np
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
sigma_values = np.arange(1, 10, 0.01)

# SNP
print('SNP')
ssim_snp = ssim_compare(image_og, image_snp)[0]
print('SSIM value original vs noised: ', ssim_snp)
ssim_snp_filter = [ssim_compare(image_og, gaussian_filter(image_snp, i+1).astype('uint8'))[0] for i in sigma_values]
snp_ssim, snp_sigma = optimal(ssim_snp_filter, sigma_values)
print('SSIM value original vs filtered:', snp_ssim)
print('Optimal sigma value: ', round(snp_sigma, 2))

# Gaussian
print('Gaussian')
ssim_gaus = ssim_compare(image_og, image_gaus)[0]
print('SSIM value original vs noised: ', ssim_gaus)
ssim_gaus_filter = [ssim_compare(image_og, gaussian_filter(image_gaus, i+1).astype('uint8'))[0] for i in sigma_values]
gaus_ssim, gaus_sigma = optimal(ssim_gaus_filter, sigma_values)
print('SSIM value original vs filtered:', gaus_ssim)
print('Optimal sigma value: ', round(gaus_sigma, 2))

'''
plt.figure()
plt.plot(sigma_values, ssim_scores)
plt.show()


ssim_val_1 = ssim_compare(image_og, image_np)
ssim_val_2 = ssim_compare(image_og, filtered_image)
print('original vs noised:', ssim_val_1)
print('original vs filtered:', ssim_val_2)

#print(f"Optimal sigma: {optimal_sigma}")
#print(f"Maximized PSNR: {max_psnr:.2f}")

plt.figure()

plt.subplot(1, 2, 1)
plt.imshow(image_og, cmap='gray')
plt.title('original')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('blurred')

plt.show()

plt.figure()
plt.imshow(blurred_image, cmap='gray')
plt.axis('off')
plt.imsave('gaus_gaus_peppers.jpg', blurred_image, cmap='gray')
'''
