import cv2
import matplotlib.pyplot as plt
# Load the image
image_path = './R.png'
# image_path = './test.jpeg'
image = cv2.imread(image_path)
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 40)
# Apply Canny Edge Detection
edges = cv2.Canny(blurred, threshold1=140, threshold2=170)
# Display the result using matplotlib
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title("Canny Edge Detection")
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()