'''
Assignment 3
Group C 
Members:
    1. 20CS10073 - Vikas Vijaykumar Bastewad
    2. 20CS30038 - Pranil Dey
'''

import cv2
import numpy as np

# Load the image of the painting
image = cv2.imread('Sheephard_Iasi.jpg')

# Define the target aspect ratios
aspect_ratios = [(2, 3), (3, 4)]

# Function to compute Homography matrices and display transformed images
def compute_homography(image, aspect_ratios):
    for aspect_ratio in aspect_ratios:
        # Define the target rectangle based on the aspect ratio
        target_width = 300  # Adjust the width as needed
        target_height = int(target_width * aspect_ratio[1] / aspect_ratio[0])
        target_rect = np.array([[0, 0], [0, target_height - 1], [target_width - 1, target_height - 1], [target_width - 1, 0]], dtype=np.float32)

        # Compute the Homography matrix
        homography_matrix, _ = cv2.findHomography(np.array([[0, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1], [image.shape[1] - 1, 0]], dtype=np.float32), target_rect)
        print("Homography matrix for aspect ratio {}x{}:".format(aspect_ratio[0], aspect_ratio[1]))
        print(homography_matrix)

        # Warp the image using the Homography matrix
        transformed_image = cv2.warpPerspective(image, homography_matrix, (target_width, target_height))

        # Display the transformed image
        cv2.imshow('Transformed Image {}x{}'.format(aspect_ratio[0], aspect_ratio[1]), transformed_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Display the original image
cv2.imshow('Original Image', image)

# Compute Homography matrices and display transformed images
compute_homography(image, aspect_ratios)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()