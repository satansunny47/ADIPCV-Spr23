'''
Assignment 3
Group C 
Members:
    1. 20CS10073 - Vikas Vijaykumar Bastewad
    2. 20CS30038 - Pranil Dey
'''

import tkinter as tk
import subprocess

# List of file names
file_names = ["Part1.py", "Part2.py", "Part3.py", "Part4.py", "Part5.py", "Part6.py"]
file_descriptions = ["Compute Pixel Coordinates", "Compute Line Length", "Identify Perpendicular Lines and Compute Transformed Dual Conic","Compute Homography Matrices for Mapping to Target Rectangles","Perform Affine Rectification","Perform Metric Rectification"]
# Function to run the selected .py file
def run_selected_file():
    selected_file = file_names[drop_var.get()]
    subprocess.Popen(['python', selected_file])

# Creating the main application window
root = tk.Tk()
root.title("Tool")


# Displaying the list of file names and its corresponding index and file description
for i in range(len(file_names)):
    label = tk.Label(root, text=f"{i}: {file_descriptions[i]}")
    label.pack()

# Label to display the instructions
label = tk.Label(root, text="Choose the index to perform required task:")
label.pack()

# Dropdown menu to select the file number
drop_var = tk.IntVar()
drop_var.set(0)  # Default selection
drop_menu = tk.OptionMenu(root, drop_var, *range(len(file_names)))
drop_menu.pack()

# Button to run the selected file
run_button = tk.Button(root, text="Run", command=run_selected_file)
run_button.pack()

# displaying the message that user can run next file by just pressing any button on keyboart
label = tk.Label(root, text="** After you have successfully run a file, Press any key if you want to select other file to run")
label.pack()

# Run the main event loop
root.mainloop()
