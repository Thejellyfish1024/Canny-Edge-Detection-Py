import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Load image
img_path = "./R.png"
ORG_RGB = mpimg.imread(img_path)

# Convert RGB to grayscale
if ORG_RGB.ndim == 3:
    GRAY_IMG = np.dot(ORG_RGB[..., :3], [0.2989, 0.5870, 0.1140])
else:
    GRAY_IMG = ORG_RGB

# Normalize grayscale if needed
if GRAY_IMG.max() > 1:
    GRAY_IMG = GRAY_IMG / 255.0

# -----------------------------
# Display Original and Grayscale Side-by-Side
# -----------------------------
plt.figure(figsize=(10, 10))

# Original RGB Image
plt.subplot(1, 2, 1)
plt.imshow(ORG_RGB)
plt.title("Original RGB Image")
plt.axis('off')

# Grayscale Image
plt.subplot(1, 2, 2)
plt.imshow(GRAY_IMG, cmap='gray', vmin=0, vmax=1)
plt.title("Grayscale Image")
plt.axis('off')

plt.show()
