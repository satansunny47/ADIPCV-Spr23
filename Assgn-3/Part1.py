'''
Assignment 3
Group C 
Members:
    1. 20CS10073 - Vikas Vijaykumar Bastewad
    2. 20CS30038 - Pranil Dey
'''

import cv2
# Loading the image
image = cv2.imread('Sheephard_Iasi.jpg')

print("Click on the image to get the pixel coordinates of the selected point.")

# Function to get pixel coordinates on mouse click
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Pixel coordinates of the selected point: ({}, {})".format(x, y))

# Display the image and wait for a mouse click
cv2.imshow('Image', image)
cv2.setMouseCallback('Image', get_coordinates)
cv2.waitKey(0)
cv2.destroyAllWindows()
