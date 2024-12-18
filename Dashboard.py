# Dashboard.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button, Checkbutton, BooleanVar, messagebox, StringVar, OptionMenu
from KMeansMethod import ImageProcessor


class Dashboard:
    def __init__(self):
        self.image_processor = None
        self.k_value = 0
        self.use_savgol = False
        self.n_neighbors = 1  # Default value for neighbors
        self.root = Tk()
        self.root.title("K-Means Clustering Dashboard")
        self.display_option = StringVar(value="Segmented Image")
        self.create_widgets()

    def create_widgets(self):
        Label(self.root, text="Select Image:").grid(row=0, column=0)
        self.image_name_var = StringVar()
        self.image_name_var.set("Select an image")
        self.image_files = self.get_image_files()
        self.image_dropdown = OptionMenu(self.root, self.image_name_var, *self.image_files)
        self.image_dropdown.grid(row=0, column=1)

        Label(self.root, text="Number of Clusters (K):").grid(row=1, column=0)
        self.k_value_entry = Entry(self.root)
        self.k_value_entry.grid(row=1, column=1)

        Label(self.root, text="Number of Neighbor Pixels (n):").grid(row=2, column=0)
        self.n_value_entry = Entry(self.root)
        self.n_value_entry.grid(row=2, column=1)

        self.slider_var = BooleanVar()
        slider_check = Checkbutton(self.root, text="Use Gaussian Blur Slider", variable=self.slider_var)
        slider_check.grid(row=3, columnspan=2)

        self.savgol_var = BooleanVar()
        savgol_check = Checkbutton(self.root, text="Use Savitzky-Golay Filter", variable=self.savgol_var)
        savgol_check.grid(row=4, columnspan=2)

        Label(self.root, text="Display Option:").grid(row=5, column=0)
        OptionMenu(self.root, self.display_option, "Segmented Image", "Border Map").grid(row=5, column=1)

        submit_button = Button(self.root, text="Submit", command=self.on_submit)
        submit_button.grid(row=6, columnspan=2)

    def get_image_files(self):
        supported_extensions = ('.jpg', '.png')
        return [f for f in os.listdir('.') if f.endswith(supported_extensions)]

    def update_images(self, blur_sigma):
        blurred_image = self.image_processor.apply_gaussian_blur(blur_sigma)
        if self.use_savgol:
            grad_image = self.image_processor.savitzky_golay(window_length=11, order=3)
        else:
            grad_image = self.image_processor.apply_canny(blurred_image)

        centroids, labels = self.image_processor.k_means_cluster(self.k_value)
        coloring = np.random.randint(0, 255, size=(self.k_value, 3))
        segmented_image = coloring[labels].reshape(self.image_processor.M, self.image_processor.N, 3).astype(np.uint8)

        n_neighbors = self.n_neighbors  # Get the value for n
        kmeans_border_map = self.image_processor.create_border_map(labels, n_neighbors)

        self.axes[0, 0].imshow(self.image_processor.image)
        self.axes[0, 0].set_title('Original Image')
        self.axes[0, 0].axis('off')

        self.axes[0, 1].imshow(grad_image, cmap='gray')
        title = 'Savitzky-Golay Filter Output' if self.use_savgol else f'Canny Edges (Blur Sigma={blur_sigma:.1f})'
        self.axes[0, 1].set_title(title)
        self.axes[0, 1].axis('off')

        if self.display_option.get() == "Segmented Image":
            self.axes[1, 0].imshow(segmented_image)
            self.axes[1, 0].set_title(f'K-means Segmentation (K={self.k_value})')
        else:
            self.axes[1, 0].imshow(kmeans_border_map, cmap='gray')
            self.axes[1, 0].set_title('K-means Border Map')
        self.axes[1, 0].axis('off')

        if self.display_option.get() == "Border Map":
            combined_borders = np.where(kmeans_border_map > 0, 1, 0)
            combined_edges = np.where(grad_image > 0, 1, 0)
            combined_segments = np.zeros_like(combined_borders)
            combined_segments[(combined_borders == 1) & (combined_edges == 1)] = 1

            self.axes[1, 1].imshow(combined_segments, cmap='gray')
            self.axes[1, 1].set_title('Combined K-means Borders and Canny Edges')
        else:
            combined_segments = np.where(grad_image[..., np.newaxis], 255, segmented_image)
            self.axes[1, 1].imshow(combined_segments)
            self.axes[1, 1].set_title('Combined Edges and Segments')

        self.axes[1, 1].axis('off')
        plt.draw()

    def on_submit(self):
        self.image_name = self.image_name_var.get()
        k_value_str = self.k_value_entry.get()
        n_value_str = self.n_value_entry.get()  # Get the n value

        try:
            self.k_value = int(k_value_str)
            if self.k_value <= 0:
                raise ValueError("K must be a positive integer.")
            self.n_neighbors = int(n_value_str)  # Convert n to int
            if self.n_neighbors < 1:
                raise ValueError("n must be a positive integer.")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        try:
            self.image_processor = ImageProcessor(self.image_name)
            blur_sigma = 2.5 if self.slider_var.get() else 0
            self.use_savgol = self.savgol_var.get()
            plt.clf()
            self.figure, self.axes = plt.subplots(2, 2, figsize=(12, 10))
            self.update_images(blur_sigma)

            if self.slider_var.get():
                ax_sigma = plt.axes([0.15, 0.01, 0.7, 0.03])
                sigma_slider = plt.Slider(ax_sigma, 'Blur Sigma', 2.0, 4.0, valinit=blur_sigma)
                sigma_slider.on_changed(lambda val: self.update_images(val))

            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run()
