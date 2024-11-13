import cv2
import numpy as np

# Load the masked image
masked_image = cv2.imread('MedSAM/assets/img_demo_mask.png', cv2.IMREAD_GRAYSCALE)

# Perform connected component analysis
num_labels, labels = cv2.connectedComponents(masked_image)

# If you have a reference object, define these variables:
# reference_object_pixels = ...  # Measure this from your image
# reference_object_real_size = ...  # Known size of your reference object
# pixels_per_unit = reference_object_pixels / reference_object_real_size

# Analyze each mask
for label in range(1, num_labels):  # Start from 1 to skip background
    mask = labels == label
    pixel_count = np.sum(mask)

    # Get mask dimensions
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    print(f"Mask {label}:")
    print(f"  Pixel count: {pixel_count}")
    print(f"  Width: {w} pixels")
    print(f"  Height: {h} pixels")

    # If you have a reference object and defined pixels_per_unit:
    # real_width = w / pixels_per_unit
    # real_height = h / pixels_per_unit
    # print(f"  Real width: {real_width:.2f} units")
    # print(f"  Real height: {real_height:.2f} units")