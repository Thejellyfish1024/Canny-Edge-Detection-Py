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
  kernel_size = kernel.shape
  img_height, img_width = img.shape #Getting image height and width from gray scale image
  convolved_image = np.zeros(img.shape)
  # Looping through each pixel of the image
  for img_row_idx in range(0,img_height):
    for img_col_idx in range(0,img_width):
        aggregate = 0
        # For each pixel, loop through full kernel to calculate aggregated pixel value
        for k_x in range(0, kernel_size[0]):
            for k_y in range(0, kernel_size[1]):
                # Getting the image index to be multiply with the kernel index
                imageBoundaryOfHeight = int(img_row_idx + (k_x - int(kernel_size[0]/2)))
                imageBoundaryOfWidth = int(img_col_idx + (k_y - int(kernel_size[0]/2)))
                # Not considering the index outside the image boundery or assuming as 0
                if imageBoundaryOfHeight>=0 and imageBoundaryOfHeight<img_height and imageBoundaryOfWidth>=0 and imageBoundaryOfWidth<img_width:
                    # extract pixel value and multiply with kernel value
                    pixel = img[imageBoundaryOfHeight, imageBoundaryOfWidth]
                    # then sum all the neighbour pixels values and store
                    aggregate = aggregate + int(pixel * kernel[k_x, k_y])
        convolved_image[img_row_idx, img_col_idx] = aggregate
  return convolved_image

# 3. Apply Gaussian Blur to remove noise-------------------
def apply_gaussian_blur(img):
  kernel_size = (5,5)
  sigma = 1
  gaussian_kernel = np.zeros(kernel_size)

  for kernel_row in range(kernel_size[0]):
    row = kernel_row - int( kernel_size[0]/2 ) # calculating Index for each kernel row keeping kernel center at (0,0) | for (3,3) kernel row (-1,1)
    for kernel_col in range(kernel_size[1]):
      col = kernel_col - int( kernel_size[1]/2 ) # Calculating Index for each kernel column keeping kernel center at (0,0) | for (3,3) kernel col (-1,1)
      # Applying formula for gussian kernel
      gaussian_kernel[kernel_row, kernel_col] = (1/(2 * math.pi * sigma**2)) * (math.e ** (-1*( (row**2 + col**2)/ (2*sigma**2) )))

  # Applying normalization on the kernel so that the sum of the kernel is <= 1. For the reason that it doesnot impact the image brightness
  kernel_sum = np.sum(gaussian_kernel)
  gaussian_kernel = gaussian_kernel / kernel_sum # will always be <=1 because, kernel pixel <= kernel_sum


  smooth_image = apply_convolution(img, gaussian_kernel) # applying convolution using gaussian kernel
  smooth_image = np.clip(smooth_image, 0, 255).astype(np.uint8) # Limiting pixel values to be 0 to 255, with data type as uint8

  return smooth_image

# 4. Apply Sobel to extract magnitude and orientation of image-------------
def apply_sobel(smooth_image):
  # Horizontal gradient convolution kernel, Detects vertical edges
  Sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
  # Vertical gradient convolution kernel, Detects horizontal edges
  Sy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
  # Applying convolution on the smooth image using both Horizontal and Vertical gradient kernels
  Gx = apply_convolution(smooth_image, Sx)
  Gy = apply_convolution(smooth_image, Sy)

  # Limiting values of Gx to 0.0001 to 255, so that Direction doesnot get invalid number
  Gx = np.clip(Gx, 0.0001, 255)
  magnitude = np.sqrt(Gx**2+Gy**2) #Applying formula for magnitude
  magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
  direction = np.arctan(Gy/Gx) #Applying formula for Direction
  return (magnitude, direction)

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
    """
    Apply double threshold to classify strong and weak edges
    """
    high = img.max() * high_ratio
    low = high * low_ratio

    strong = 255
    weak = 75

    strong_edges = (img >= high).astype(np.uint8) * strong
    weak_edges = ((img >= low) & (img < high)).astype(np.uint8) * weak

    return strong_edges, weak_edges

# 7. Apply Edge Tracking by Hysteresis to get the final image----------------------
def apply_hysteresis(strong, weak):
    """
    Connect weak edges to strong edges if neighbors are strong
    """
    img_h, img_w = strong.shape
    final_img = np.copy(strong)
    
    # Directions (8 neighbors)
    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, -1, 1, -1, 0, 1]

    # Process weak edges
    for i in range(1, img_h-1):
        for j in range(1, img_w-1):
            if weak[i,j] != 0:
                # Check 8 neighbors
                if any(final_img[i+dx[k], j+dy[k]] == 255 for k in range(8)):
                    final_img[i,j] = 255
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
