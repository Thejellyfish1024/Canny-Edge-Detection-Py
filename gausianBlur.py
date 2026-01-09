import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

# Load image
img_path = "./R.png"
# img_path = "./test.jpeg"
# img_path = "./test2.jpeg"
ORG_RGB = mpimg.imread(img_path)

# Convert RGB to grayscale
def convert_to_grayscale(img):
  img_height, img_width, channels = img.shape ##Getting image height and width from RGB image
  gray_scale_image = np.zeros((img_height, img_width))
  # Looping through each pixel of the image
  for img_row_idx in range(0,img_height):
    for img_col_idx in range(0,img_width):
      # Converting to standard python int type to uint8 type values. so that Data is not lost in the sum
      r = int(img[img_row_idx, img_col_idx, 0])
      g = int(img[img_row_idx, img_col_idx, 1])
      b = int(img[img_row_idx, img_col_idx, 2])
      sum = r+g+b
      average = sum/3 # Applying Average method for gray scale conversion
      gray_scale_image[img_row_idx, img_col_idx] = average
  return gray_scale_image
# -----------------------------
# Convolution Process
def apply_convolution(img, kernel):
    """
    Apply a convolution between an image and a kernel
    """
    # Get image and kernel dimensions
    img_h, img_w = img.shape
    k_h, k_w = kernel.shape

    # Calculate padding
    pad_h = k_h // 2
    pad_w = k_w // 2

    # Pad image with zeros
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Prepare output image
    output = np.zeros_like(img)
    
    # Convolution operation
    for i in range(img_h):
        for j in range(img_w):
            # Extract the current region
            region = padded_img[i:i+k_h, j:j+k_w]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)
    
    return output
# -----------------------------
# Apply Gaussian Blur to remove noise
def apply_gaussian_blur(img, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian Blur to remove noise
    """
    # Create Gaussian Kernel
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)  # Normalize

    # Apply convolution
    blurred_img = apply_convolution(img, kernel)
    return blurred_img


# -----------------------------
# Display Original and Grayscale Side-by-Side
# -----------------------------
gray_scale_image = convert_to_grayscale(ORG_RGB)
smooth_image = apply_gaussian_blur(gray_scale_image)

plt.figure(figsize=(10, 10))

# Original RGB Image
plt.subplot(1, 2, 1)
plt.imshow(gray_scale_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

# Grayscale Image
plt.subplot(1, 2, 2)
plt.imshow(smooth_image, cmap='gray')
plt.title("Smooth Image")
plt.axis('off')

plt.show()
