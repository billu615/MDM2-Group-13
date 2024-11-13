from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import cv2

def select_image(name):
    image = Image.open('initial/'+name+'_gray.jpg').convert('L')
    image_og = np.array(image)
    image = Image.open('initial/n_gaus_'+name+'.jpg').convert('L')
    image_gaus = np.array(image)
    image_noise = Image.open('initial/n_snp_'+name+'.jpg').convert('L')
    image_snp = np.array(image_noise)
    return image_og, image_gaus, image_snp

# Fourier
def four(image, radius_value):
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    radius = radius_value
    cv2.circle(mask, (ccol, crow), radius, 1, -1)
    filtered_shifted = f_transform_shifted * mask
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_shifted))
    filtered_image = np.abs(filtered_image).astype('uint8')
    return filtered_image

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
radii = np.arange(5, 100)

# SNP
print('SNP')
ssim_snp = ssim_compare(image_og, image_snp)[0]
print('SSIM value original vs noised: ', ssim_snp)
ssim_snp_filter = [ssim_compare(image_og, four(image_snp, i+1))[0] for i in radii]
snp_ssim, snp_radius = optimal(ssim_snp_filter, radii)
print('SSIM value original vs filtered:', snp_ssim)
print('Optimal radius: ', snp_radius)

# Gaussian
print('Gaussian')
ssim_gaus = ssim_compare(image_og, image_gaus)[0]
print('SSIM value original vs noised: ', ssim_gaus)
ssim_gaus_filter = [ssim_compare(image_og, four(image_gaus, i+1))[0] for i in radii]
snp_ssim, snp_radius = optimal(ssim_gaus_filter, radii)
print('SSIM value original vs filtered:', snp_ssim)
print('Optimal radius: ', snp_radius)

# Format [[lenna], [peppers], [circle]] within each list [original, sg, gaussian, fourier]
snp_array = [[0.073836831739095, 0.5401511035084902,  0.5782332430881044, 0.5361800502994166],
             [0.05409969631503954, 0.6359022815449027, 0.6796118766296892, 0.636851727052561],
             [0.009758354968389917, 0.2870037754046791, 0.29386641292439, 0.28732648822170564]]

gaus_array = [[0.17663920770574273, 0.6366914298352679, 0.6770316338127695, 0.60270152917455],
              [0.12853877851410303, 0.7154663000157111, 0.764001458299605, 0.6902355216051212],
              [0.024806908263132407, 0.32312918815739544, 0.3297707732729077, 0.3195955534218667]]

'''
plt.figure()
plt.plot(radius, ssim_scores)
plt.show()


plt.figure()
plt.subplot(1, 2, 1)
plt.title("Original Image", fontsize = 12)
plt.imshow(image, cmap='gray')
plt.axis('off')


plt.subplot(1, 2, 2)
plt.title("Filtered Image (Low-Pass)", fontsize = 12)
plt.imshow(filtered_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
'''
