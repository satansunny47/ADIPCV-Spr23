import csv
import matplotlib.pyplot as plt
import numpy as np

# Reading data from the csv file
def read_xyz_data(file_path):
    xyz_data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            wavelength, x, y, z = map(float, row)
            xyz_data.append((wavelength, x, y, z))
    return xyz_data

# Convert RGB values to XYZ color space
def rgb_to_xyz(rgb_values):
    r, g, b = rgb_values
    r /= 255.0
    g /= 255.0
    b /= 255.0

    # Applying the sRGB gamma correction
    r = pow((r + 0.055) / 1.055, 2.4) if r > 0.04045 else r / 12.92
    g = pow((g + 0.055) / 1.055, 2.4) if g > 0.04045 else g / 12.92
    b = pow((b + 0.055) / 1.055, 2.4) if b > 0.04045 else b / 12.92

    # Convert RGB to XYZ
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    return x, y, z

# Normalize XYZ values to obtain chromaticity coordinates
def normalize_xyz(xyz_values):
    x, y, z = xyz_values
    sum_xyz = x + y + z
    return x / sum_xyz, y / sum_xyz

# Load the original, saturated, and desaturated images and data from csv
original_image_path = "flower.jpg"
saturated_image_path = "maximally_saturated_image.jpg"
desaturated_image_path = "desaturated_image.jpg"

xyz_data_path = "ciexyz31_1.csv"
xyz_data = read_xyz_data(xyz_data_path)

# Initialize lists to store chromaticity coordinates
original_chromaticity_points = []
saturated_chromaticity_points = []
desaturated_chromaticity_points = []

# Process each image
for image_path, chromaticity_points in zip([original_image_path, saturated_image_path, desaturated_image_path],
                                           [original_chromaticity_points, saturated_chromaticity_points, desaturated_chromaticity_points]):

    image = plt.imread(image_path)

    # Convert RGB values to XYZ color space and normalize XYZ values
    for row in image:
        for pixel in row:
            xyz_values = rgb_to_xyz(pixel[:3])  # Extract RGB values
            chromaticity = normalize_xyz(xyz_values)
            chromaticity_points.append(chromaticity)

# Convert lists to NumPy arrays for easier manipulation
original_chromaticity_points = np.array(original_chromaticity_points)
saturated_chromaticity_points = np.array(saturated_chromaticity_points)
desaturated_chromaticity_points = np.array(desaturated_chromaticity_points)

# Plot the chromaticity points
plt.figure(figsize=(8, 6))
plt.scatter(original_chromaticity_points[:, 0], original_chromaticity_points[:, 1], s=1, c='r', label='Original Image')
plt.scatter(saturated_chromaticity_points[:, 0], saturated_chromaticity_points[:, 1], s=1, c='g', label='Saturated Image')
plt.scatter(desaturated_chromaticity_points[:, 0], desaturated_chromaticity_points[:, 1], s=1, c='b', label='Desaturated Image')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Chromaticity Points')
plt.legend()

plt.savefig('chromaticity_points_plot.png')
plt.show()




