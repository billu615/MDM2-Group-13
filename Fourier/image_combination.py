import matplotlib.pyplot as plt
import matplotlib.image as mpimg


path = ''
image_names = [
    'lenna.jpg', 'n_snp_lenna.jpg', 'n_gaus_lenna.jpg',
    'canny_lenna.jpg', 'canny_snp_lenna.jpg', 'canny_gaus_lenna.jpg',
    'lp_canny_lenna.jpg', 'lp_canny_snp_lenna.jpg', 'lp_canny_snp_lenna.jpg'
]

images = [mpimg.imread(path + name) for name in image_names]

fig, axes = plt.subplots(3, 3, figsize=(8, 8))

for ax in axes.flatten():
    ax.axis('off')

for i, ax in enumerate(axes.flatten()):
    ax.imshow(images[i], cmap='gray')  # 假设图片是灰度图，如果是彩色图则移除 cmap='gray'


fig.text(0.25, 0.92, "Lenna", ha='center', fontsize=12)
fig.text(0.5, 0.92, "Salt and Pepper Noise", ha='center', fontsize=12)
fig.text(0.75, 0.92, "Gaussian Noise", ha='center', fontsize=12)


fig.text(0.08, 0.75, "Original Image", va='center', rotation='vertical', fontsize=12)
fig.text(0.08, 0.5, "Traditional Canny", va='center', rotation='vertical', fontsize=12)
fig.text(0.08, 0.25, "Canny After Fourier", va='center', rotation='vertical', fontsize=12)


plt.subplots_adjust(wspace=0.1, hspace=0.1)

output_path = 'canny_fourier_result.jpg'
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.show()

plt.show()
