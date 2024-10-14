import matplotlib.pyplot as plt
import numpy as np
from skimage import io

image = io.imread("Lena_image.jpg")
image_rgb = image[:, :, ::-1]

R = image_rgb[:, :, 0]
G = image_rgb[:, :, 1]
B = image_rgb[:, :, 2]

R_flat = R.flatten() #turns into array
G_flat = G.flatten()
B_flat = B.flatten()

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')


ax.scatter(R_flat, G_flat, B_flat, c=np.stack((R_flat, G_flat, B_flat), axis=1) / 255.0, marker='o') #turns each value into a value within [0,1]

ax.set_xlabel('Red Channel')
ax.set_ylabel('Green Channel')
ax.set_zlabel('Blue Channel')

plt.show() #bit laggy

#print(R_flat, G_flat, B_flat)