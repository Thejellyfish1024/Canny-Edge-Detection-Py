# canny.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# -----------------------------
# 1. Load Image
# -----------------------------
# Replace 'image.jpg' with your local image file path
img_path = "./test.jpeg"
# img_path = "./R.png"

ORG_IMG = mpimg.imread(img_path)

# Convert RGB to Grayscale if needed
if ORG_IMG.ndim == 3:
    ORG_IMG = np.dot(ORG_IMG[..., :3], [0.2989, 0.5870, 0.1140])

# -----------------------------
# 2. Convolution Function
# -----------------------------
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
# 3. Gaussian Blur
# -----------------------------
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
# 4. Sobel Operator
# -----------------------------
def apply_sobel(img):
    """
    Calculate gradient magnitude and direction using Sobel
    """
    # Sobel Kernels
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    # Apply convolution
    Gx = apply_convolution(img, Kx)
    Gy = apply_convolution(img, Ky)

    # Gradient magnitude
    magnitude = np.hypot(Gx, Gy)
    magnitude = magnitude / magnitude.max() * 255  # Normalize to 0-255

    # Gradient direction
    theta = np.arctan2(Gy, Gx)
    
    return magnitude, theta

# -----------------------------
# 5. Non-Maximum Suppression
# -----------------------------
def apply_nonmaximum_suppression(magnitude, theta):
    """
    Thin edges using Non-Maximum Suppression
    """
    img_h, img_w = magnitude.shape
    Z = np.zeros((img_h, img_w), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, img_h-1):
        for j in range(1, img_w-1):
            q = 255
            r = 255
            # Angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # Angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # Angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # Angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]
            
            if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                Z[i,j] = magnitude[i,j]
            else:
                Z[i,j] = 0
    return Z

# -----------------------------
# 6. Double Threshold
# -----------------------------
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

# -----------------------------
# 7. Edge Tracking by Hysteresis
# -----------------------------
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

# -----------------------------
# 8. Run Full Canny Edge Detection
# -----------------------------
blurred_img = apply_gaussian_blur(ORG_IMG)
gradient_magnitude, gradient_direction = apply_sobel(blurred_img)
thin_edges = apply_nonmaximum_suppression(gradient_magnitude, gradient_direction)
strong_edges, weak_edges = apply_double_thresholding(thin_edges)
FINAL_IMG = apply_hysteresis(strong_edges, weak_edges)

# -----------------------------
# 9. Display Side-by-Side
# -----------------------------
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(ORG_IMG, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(FINAL_IMG, cmap='gray')
plt.title("Edge Detected Image")
plt.axis('off')
plt.show()



