import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

class PointRecorder:
    def __init__(self, master, image1_path, image2_path):
        self.master = master
        self.image1 = Image.open(image1_path)
        self.image2 = Image.open(image2_path)
        self.canvas1 = tk.Canvas(master, width=self.image1.width, height=self.image1.height)
        self.canvas2 = tk.Canvas(master, width=self.image2.width, height=self.image2.height)
        self.canvas1.pack(side=tk.LEFT)
        self.canvas2.pack(side=tk.RIGHT)
        self.points_image1 = []
        self.points_image2 = []
        self.photo1 = ImageTk.PhotoImage(self.image1)
        self.photo2 = ImageTk.PhotoImage(self.image2)
        self.canvas1.create_image(0, 0, image=self.photo1, anchor=tk.NW)
        self.canvas2.create_image(0, 0, image=self.photo2, anchor=tk.NW)
        self.canvas1.bind("<Button-1>", self.record_point_image1)
        self.canvas2.bind("<Button-1>", self.record_point_image2)
    def record_point_image1(self, event):
        self.points_image1.append((event.x, event.y))
        self.canvas1.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill='red')

    def record_point_image2(self, event):
        self.points_image2.append((event.x, event.y))
        self.canvas2.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill='red')

if __name__ == "__main__":
    root = tk.Tk()
    app = PointRecorder(root, "Baltimore_A1.jpg", "Baltimore_A2.jpg")
    print("Select 8 corresponding points on each of the two images.\n")
    root.mainloop()

# Convert the lists of points to NumPy arrays
points_image1 = np.array(app.points_image1)
points_image2 = np.array(app.points_image2)

# Calculate the homography matrix H using OpenCV
H, _ = cv2.findHomography(points_image1, points_image2)
print("Homography matrix:\n")
print(H)
print("\n")

# Given calibration matrix K
K = np.array([[-288, 0, 288], [0, 512, 512], [0, 0, 1]])

# Estimate the rotation matrix R from H and K
R = np.linalg.inv(K) @ H
print("Rotation matrix:\n")
print(R)
print("\n")

# Use the method of 8-point correspondences to estimate the fundamental matrix
F, _ = cv2.findFundamentalMat(points_image1, points_image2,cv2.FM_8POINT)
print("Fundamental matrix:\n")
print(F)
