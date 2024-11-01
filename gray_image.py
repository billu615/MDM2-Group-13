from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def image_to_rgb(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    width, height = img.size
    image_list = []
    for i in range(height):
        row = []
        for j in range(width):
            r, g, b = img.getpixel((j, i))
            row.append((r, g, b))
        image_list.append(row)
    return image_list

def rgb_to_gray(image):
    height = len(image)
    width = len(image[0])
    gray_image = [[0 for _ in range(width)] for _ in range(height)]  # Corrected here
    for i in range(height):
        for j in range(width):
            r, g, b = image[i][j]
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)  # Conversion to grayscale
            gray_image[i][j] = gray
    return gray_image

image_path = 'car.jpg'
image = image_to_rgb(image_path)  # Corrected here
gray = rgb_to_gray(image)