import numpy as np
import argparse
import cv2
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Function to perform colorization


def colorize_image():
    image_path = image_path_var.get()

    # Load the Model
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)

    # Load centers for ab channel quantization used for rebalancing.
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Load the input image
    image = cv2.imread(image_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Colorizing the image
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Display the colorized image
    colorized_image = Image.fromarray(colorized)
    colorized_image = ImageTk.PhotoImage(image=colorized_image)
    result_label.config(image=colorized_image)
    result_label.image = colorized_image


# Create the main window
root = tk.Tk()
root.title("Image Colorization")

# File selection button
file_button = tk.Button(root, text="Select Image",
                        command=lambda: image_path_var.set(filedialog.askopenfilename()))
file_button.pack(pady=10)

# Colorize button
colorize_button = tk.Button(root, text="Colorize", command=colorize_image)
colorize_button.pack()

# Result image label
result_label = tk.Label(root)
result_label.pack()

# Default file path
image_path_var = tk.StringVar()
image_path_var.set("")

# Constants for file paths
DIR = r"E:\pythonprojectschlng\colorization"
PROTOTXT = os.path.join(DIR, r"colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"pts_in_hull.npy")
MODEL = os.path.join(DIR, r"colorization_release_v2.caffemodel")

root.mainloop()
