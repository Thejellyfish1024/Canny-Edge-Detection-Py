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
# Function to apply double thresholding on a grayscale image
def apply_double_thresholding(img, low_ratio=0.2, high_ratio=0.30):

    # Get the height (rows) and width (columns) of the image
    img_h, img_w = img.shape

    # Calculate the high threshold value as a percentage of the maximum pixel intensity
    high = img.max() * high_ratio

    # Calculate the low threshold value as a percentage of the high threshold
    low = high * low_ratio

    # Define the pixel value used to mark strong edges
    strong = 255

    # Define the pixel value used to mark weak edges
    weak = 75

    # Create an empty image to store strong edge pixels
    strong_edges = np.zeros((img_h, img_w), dtype=np.uint8)

    # Create an empty image to store weak edge pixels
    weak_edges = np.zeros((img_h, img_w), dtype=np.uint8)

    # Loop through each row of the image
    for i in range(img_h):

        # Loop through each column of the image
        for j in range(img_w):

            # Read the intensity value of the current pixel
            pixel = img[i, j]

            # Check if the pixel intensity is greater than or equal to the high threshold
            # If true, mark it as a strong edge
            if pixel >= high:
                strong_edges[i, j] = strong
            else:
                # Otherwise, mark it as a non-edge (0)
                strong_edges[i, j] = 0

            # Check if the pixel intensity lies between the low and high thresholds
            # Such pixels are considered weak edges
            if low <= pixel < high:
                weak_edges[i, j] = weak
            else:
                # Otherwise, mark it as a non-edge (0)
                weak_edges[i, j] = 0

    # Return both the strong edge image and the weak edge image
    return strong_edges, weak_edges



# 7. Apply Edge Tracking by Hysteresis to get the final image----------------------
# Function to apply edge tracking by hysteresis
def apply_hysteresis(strong, weak):

    # Get the height (rows) and width (columns) of the image
    img_h, img_w = strong.shape

    # Create the final edge image by copying the strong edge image
    # Strong edges are always preserved
    final_img = np.copy(strong)

    # Define relative positions of the 8-connected neighboring pixels
    # These cover all directions around a pixel (N, S, E, W, and diagonals)
    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1,  0,  1, -1, 1, -1, 0, 1]

    # Loop through the image excluding border pixels
    # Border pixels are skipped to avoid out-of-bounds access
    for i in range(1, img_h - 1):

        # Loop through each column excluding borders
        for j in range(1, img_w - 1):

            # Process only pixels that are classified as weak edges
            if weak[i, j] != 0:

                # Flag to check whether the weak edge is connected to a strong edge
                connected_to_strong = False

                # Check all 8 neighboring pixels
                for k in range(8):

                    # Compute the row index of the neighboring pixel
                    ni = i + dx[k]

                    # Compute the column index of the neighboring pixel
                    nj = j + dy[k]

                    # If any neighboring pixel is a strong edge
                    if final_img[ni, nj] == 255:
                        # Mark this weak pixel as connected to a strong edge
                        connected_to_strong = True
                        # Stop checking further neighbors
                        break

                # If the weak edge is connected to at least one strong edge
                if connected_to_strong:
                    # Promote the weak edge to a strong edge
                    final_img[i, j] = 255
                else:
                    # Otherwise, suppress the weak edge (remove it)
                    final_img[i, j] = 0

    # Return the final edge-detected image after hysteresis
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
