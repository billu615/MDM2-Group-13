import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = {
    'Image': ['Circle', 'Circle', 'Circle', 'Circle', 'Peppers', 'Peppers', 'Peppers', 'Peppers', 'Lenna', 'Lenna', 'Lenna', 'Lenna'],
    'Values': [10, 15, 20, 25, 15, 10, 25, 20, 30, 35, 20, 30],
    'Subcategory': ['Noisy Image', 'Gaussian', 'SG Filter', 'Forrier', 'Noisy Image', 'Gaussian', 'SG Filter', 'Forrier', 'Noisy Image', 'Gaussian', 'SG Filter', 'Forrier']
}

df = pd.DataFrame(data)

# Create a barplot with hue
sns.barplot(x='Image', y='Values', hue='Subcategory', data=df)

# Add title and labels
plt.title('Barplot of Categories and Values with Subcategories')
plt.xlabel('Image')
plt.ylabel('SSIM Value')
plt.savefig('image_SSIM.jpg')
# Show the plot
plt.show()


