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
        self.points = {'image1': [], 'image2': []}
        self.photo1 = ImageTk.PhotoImage(self.image1)
        self.photo2 = ImageTk.PhotoImage(self.image2)
        self.canvas1.create_image(0, 0, image=self.photo1, anchor=tk.NW)
        self.canvas2.create_image(0, 0, image=self.photo2, anchor=tk.NW)
        self.canvas1.bind("<Button-1>", self.record_point)
        self.canvas2.bind("<Button-1>", self.record_point)

    def record_point(self, event):
        if event.widget == self.canvas1:
            self.points['image1'].append((event.x, event.y))
            self.canvas1.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill='red')
            print("Image 1: ", event.x, event.y)
        elif event.widget == self.canvas2:
            self.points['image2'].append((event.x, event.y))
            self.canvas2.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill='red')
            print("Image 2: ", event.x, event.y)

if __name__ == "__main__":
    root = tk.Tk()
    app = PointRecorder(root, "Baltimore_A1.jpg", "Baltimore_A2.jpg")
    print("You can start selecting points now :) \n")
    root.mainloop()