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

# Function to identify perpendicular lines and compute the transformed dual conic at infinity
def compute_dual_conic(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    
    # Initialize a list to store the detected perpendicular lines
    perpendicular_lines = []
    
    # Loop through the detected lines to find perpendicular pairs
    for line1 in lines:
        for line2 in lines:
            if line1 is not line2:
                rho1, theta1 = line1[0]
                rho2, theta2 = line2[0]
                angle_diff = np.abs(theta1 - theta2)
                if np.abs(angle_diff - np.pi/2) < np.pi/18:  # Allowing a small deviation for perpendicular lines
                    perpendicular_lines.append((rho1, theta1, rho2, theta2))
    
    # Compute the transformed dual conic at infinity using the perpendicular lines
    transformed_dual_conic = np.zeros((3, 3))
    for line in perpendicular_lines:
        rho1, theta1, rho2, theta2 = line
        A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
        b = np.array([-rho1, -rho2])
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        dual_conic = np.outer(x, x)
        
        # Ensure the dimensions match for matrix addition
        dual_conic = np.pad(dual_conic, ((0, 1), (0, 1)), mode='constant')  # Pad to match the shape of transformed_dual_conic
        transformed_dual_conic += dual_conic

    return transformed_dual_conic

# Compute the transformed dual conic at infinity
dual_conic_at_infinity = compute_dual_conic(image)
print("Transformed Dual Conic at Infinity:")
print(dual_conic_at_infinity)
