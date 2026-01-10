# canny.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

# -----------------------------
# 1. Load Image
# -----------------------------
# Replace 'image.jpg' with your local image file path
img_path = "./R.png"
ORG_IMG = mpimg.imread(img_path)


# 1. Gray Scale Conversion------------------
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

# 2. Convolution Process------------------------
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
# 3. Apply Gaussian Blur to remove noise-------------------
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

# 4. Apply Sobel to extract magnitude and orientation of image-------------
def apply_sobel(smooth_image):
    # Sobel kernels
    Sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    Sy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    # Convolution
    Gx = apply_convolution(smooth_image, Sx)
    Gy = apply_convolution(smooth_image, Sy)

    # DO NOT CLIP gradients ❌
    # Compute magnitude (keep float)
    magnitude = np.sqrt(Gx**2 + Gy**2)

    # Compute direction (correct way)
    direction = np.arctan2(Gy, Gx)

    return magnitude, direction

# 5. Apply Non-Maximum Suppression to extract thik and thin edge-----------------
def apply_nonmaximum_suppression(magnitude, direction):
  rows, cols = magnitude.shape
  nms_image = np.zeros((rows, cols), dtype=np.float32)
  angle = direction * 180. / np.pi
  angle[angle < 0] += 180
  for i in range(1, rows - 1):
      for j in range(1, cols - 1):
          q = 255
          r = 255

          # Identify the gradient direction sector (0, 45, 90, or 135 degrees)

          # Angle 0 (Horizontal gradient -> Vertical Edge)
          # Check left and right pixels
          if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
              q = magnitude[i, j+1]
              r = magnitude[i, j-1]

          # Angle 45 (Diagonal gradient -> Diagonal Edge /)
          # Check diagonal neighbors (top-right and bottom-left)
          elif (22.5 <= angle[i,j] < 67.5):
              q = magnitude[i+1, j-1]
              r = magnitude[i-1, j+1]

          # Angle 90 (Vertical gradient -> Horizontal Edge)
          # Check top and bottom pixels
          elif (67.5 <= angle[i,j] < 112.5):
              q = magnitude[i+1, j]
              r = magnitude[i-1, j]

          # Angle 135 (Diagonal gradient -> Diagonal Edge \)
          # Check diagonal neighbors (top-left and bottom-right)
          elif (112.5 <= angle[i,j] < 157.5):
              q = magnitude[i-1, j-1]
              r = magnitude[i+1, j+1]

          # Suppression
          # If the current pixel is greater than or equal to its neighbors along the gradient, keep it.
          # Otherwise, suppress it (set to 0).
          if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
              nms_image[i,j] = magnitude[i,j]
          else:
              nms_image[i,j] = 0
  return nms_image


# 6. Double Thresholding---------------------------------
def apply_double_thresholding(img, low_ratio=0.05, high_ratio=0.15):
    img_h, img_w = img.shape

    high = img.max() * high_ratio
    low = high * low_ratio

    strong = 255
    weak = 75

    strong_edges = np.zeros((img_h, img_w), dtype=np.uint8)
    weak_edges = np.zeros((img_h, img_w), dtype=np.uint8)

    for i in range(img_h):
        for j in range(img_w):
            pixel = img[i, j]

            # Strong edge
            if pixel >= high:
                strong_edges[i, j] = strong
            else:
                strong_edges[i, j] = 0

            # Weak edge
            if low <= pixel < high:
                weak_edges[i, j] = weak
            else:
                weak_edges[i, j] = 0

    return strong_edges, weak_edges


# 7. Apply Edge Tracking by Hysteresis to get the final image----------------------
def apply_hysteresis(strong, weak):
    img_h, img_w = strong.shape
    final_img = np.copy(strong)

    # 8-connected neighbors
    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1,  0,  1, -1, 1, -1, 0, 1]

    for i in range(1, img_h - 1):
        for j in range(1, img_w - 1):

            # Process only weak edges
            if weak[i, j] != 0:
                connected_to_strong = False

                for k in range(8):
                    ni = i + dx[k]
                    nj = j + dy[k]

                    if final_img[ni, nj] == 255:
                        connected_to_strong = True
                        break

                if connected_to_strong:
                    final_img[i, j] = 255
                else:
                    final_img[i, j] = 0

    return final_img



#  Run Full Canny Edge Detection -----------------------------
gray_image = convert_to_grayscale(ORG_IMG)
smooth_image = apply_gaussian_blur(gray_image)
magnitude, direction = apply_sobel(smooth_image)
thin_edges = apply_nonmaximum_suppression(magnitude, direction)
strong_edges, weak_edges = apply_double_thresholding(thin_edges)
FINAL_IMG = apply_hysteresis(strong_edges, weak_edges)


# Show output (side-by-side)-------------------------------------
plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)
plt.imshow(ORG_IMG, cmap='gray')
plt.title("Original RGB Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(FINAL_IMG, cmap='gray')
plt.title("Edge Detected Image")
plt.axis('off')

plt.show()
