'''
Assignment 3
Group C 
Members:
    1. 20CS10073 - Vikas Vijaykumar Bastewad
    2. 20CS30038 - Pranil Dey
'''

import cv2
import numpy as np

# Global variables to store the selected points
points = []

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Function to get mouse click coordinates
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 2:
            # Calculate the distance between the two points
            distance = calculate_distance(points[0], points[1])
            print("Distance between the two points: {:.2f} pixels".format(distance))
            # Draw a line between the two points
            cv2.line(image, points[0], points[1], (255, 0, 0), 2)
            cv2.imshow('Image', image)
            points.clear()

# Load the image
image = cv2.imread('Sheephard_Iasi.jpg')

# Display the image and wait for mouse clicks
cv2.imshow('Image', image)
cv2.setMouseCallback('Image', get_coordinates)
cv2.waitKey(0)
cv2.destroyAllWindows()
