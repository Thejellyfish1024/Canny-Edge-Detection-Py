import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Load image
# img_path = "./R.png"
# img_path = "./test.jpeg"
img_path = "./test2.jpeg"
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
# Display Original and Grayscale Side-by-Side
# -----------------------------
GRAY_IMG = convert_to_grayscale(ORG_RGB)
plt.figure(figsize=(10, 10))

# Original RGB Image
plt.subplot(1, 2, 1)
plt.imshow(ORG_RGB)
plt.title("Original RGB Image")
plt.axis('off')

# Grayscale Image
plt.subplot(1, 2, 2)
plt.imshow(GRAY_IMG, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

plt.show()
