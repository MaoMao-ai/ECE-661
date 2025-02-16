import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

# Load the selected image
image_path = "cat.jpg"  # Replace with your image path
image = Image.open(image_path).convert("L")  # Convert to grayscale
image_array = np.array(image)  # Convert to a NumPy array

# Define the 3x3 convolution kernel
kernel = np.array([
    [ 0,  -1/2,  0],
    [-1/2,  1,  -1/2],
    [ 0,  -1/2,  0]
])

# Perform 2D convolution
filtered_image = convolve2d(image_array, kernel, mode='same', boundary='fill', fillvalue=0)

# Normalize the result to the range of 0-255
filtered_image = filtered_image - filtered_image.min()  # Shift minimum value to 0
filtered_image = (filtered_image / filtered_image.max()) * 255  # Normalize to 0-255
filtered_image = filtered_image.astype(np.uint8)

# Display the original and processed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_array, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title("Filtered Image")
plt.axis("off")

plt.show()

