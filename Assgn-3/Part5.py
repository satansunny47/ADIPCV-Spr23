'''
Assignment 3
Group C 
Members:
    1. 20CS10073 - Vikas Vijaykumar Bastewad
    2. 20CS30038 - Pranil Dey
'''

import cv2
import numpy as np

# Load the image
image = cv2.imread('Sheephard_Iasi.jpg')

# show the original image
cv2.imshow('Original Image', image)

# Define the coordinates of the painting edges
src_pts = np.float32([(120, 82), (363, 84), (131, 210), (354, 212)])

# Define the desired coordinates for the rectified image
dst_pts = np.float32([[0, 0], [image.shape[1], 0], [0, image.shape[0]], [image.shape[1], image.shape[0]]])

# Generate the transformation matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply the transformation to rectify the image
rectified_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

# Display the rectified image
cv2.imshow('Rectified Image', rectified_image)
cv2.waitKey(0)
cv2.destroyAllWindows()