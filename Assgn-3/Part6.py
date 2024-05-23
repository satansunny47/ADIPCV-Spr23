'''
Assignment 3
Group C 
Members:
    1. 20CS10073 - Vikas Vijaykumar Bastewad
    2. 20CS30038 - Pranil Dey
'''

import cv2
import numpy as np

# Load the image for metric rectification
image = cv2.imread('Sheephard_Iasi.jpg')

# Define corresponding points for homography calculation
# Replace these points with your selected corresponding points
pts_src = np.array([[8, 0], [475, 0], [56, 286], [432, 286]])
pts_dst = np.array([[0, 0], [510, 0], [0, 286], [510, 286]])

# Compute the homography matrix
homography_matrix, _ = cv2.findHomography(pts_src, pts_dst)

# Apply the homography transformation to rectify the image metrically
rectified_image = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))

# Compute the true aspect ratio from the transformed image
aspect_ratio = rectified_image.shape[1] / rectified_image.shape[0]

# Display the rectified image and the true aspect ratio
cv2.imshow('Rectified Image', rectified_image)
print("True Aspect Ratio: {:.2f}".format(aspect_ratio))

cv2.waitKey(0)
cv2.destroyAllWindows()
