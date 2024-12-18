import cv2

image_path1 = 'lp_lenna.jpg'
lenna_fourier_canny = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)

blurred1 =cv2.GaussianBlur(lenna_fourier_canny, (5,5), 1.4)
edges = cv2.Canny(blurred1, 50, 130)

cv2.imwrite('lp_canny_lenna.jpg', edges)



image_path2 = 'lp_gaus_lenna.jpg'
lenna_gaus_fourier_canny = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

blurred2 =cv2.GaussianBlur(lenna_gaus_fourier_canny, (5,5), 1.4)
edges = cv2.Canny(blurred3, 50, 130)

cv2.imwrite('lp_canny_gaus_lenna.jpg', edges)



image_path3 = 'lp_snp_lenna.jpg'
lenna_snp_fourier_canny = cv2.imread(image_path3, cv2.IMREAD_GRAYSCALE)

blurred3 =cv2.GaussianBlur(lenna_snp_fourier_canny, (5,5), 1.4)
edges = cv2.Canny(blurred3, 50, 130)

cv2.imwrite('lp_canny_snp_lenna.jpg', edges)



#traditional without fourier
#lenna
image_path4 = 'lenna.jpg'
lenna_canny = cv2.imread(image_path4, cv2.IMREAD_GRAYSCALE)

blurred4 =cv2.GaussianBlur(lenna_canny, (5,5), 1.4)
edges = cv2.Canny(blurred4, 50, 180)

cv2.imwrite('canny_lenna.jpg', edges)

#lenna_gaus
image_path5 = 'n_gaus_lenna.jpg'
lenna_gaus_canny = cv2.imread(image_path5, cv2.IMREAD_GRAYSCALE)

blurred5 =cv2.GaussianBlur(lenna_gaus_canny, (5,5), 1.4)
edges = cv2.Canny(blurred5, 100, 255)

cv2.imwrite('canny_gaus_lenna.jpg', edges)

#lenna_snp
image_path6 = 'n_snp_lenna.jpg'
lenna_snp_canny = cv2.imread(image_path6, cv2.IMREAD_GRAYSCALE)

blurred6 =cv2.GaussianBlur(lenna_snp_canny, (5,5), 1.4)
edges = cv2.Canny(blurred6, 190, 300)

cv2.imwrite('canny_snp_lenna.jpg', edges)




