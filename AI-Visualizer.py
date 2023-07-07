import NeuralNetwork as NN
import numpy as np

import gui
import tkinter as tk
from PIL import ImageTk, Image, ImageDraw

network = NN.Network([2,3,2])

# Open the file
with open('dataset.txt', 'r') as file:
    lines = file.readlines()
# Remove empty lines and strip whitespace
lines = [line.strip() for line in lines if line.strip()]
# Create an empty list to store datapoints
datapoints = []

# Process each line and create Datapoint objects
for line in lines:
    # Split the line by comma and convert values to floats
    values = [float(val) for val in line.split(',')]
    # Extract expected output and inputs
    expected_output = values[:2]
    inputs = values[2:]
    # Create a Datapoint object and append it to the list
    datapoint = NN.Datapoint(expected_output, inputs)
    datapoints.append(datapoint)


def draw_network(x1, y1, x2, y2, datapoints:list[NN.Datapoint]):
    while network.AverageCost(datapoints) > 0.5:
        network.Learn(datapoints, 0.03)
        print(network.AverageCost(datapoints))
        increment = 2
        radius = 5
        xSize = x2-x1
        ySize = y2-y1
        pixelIncrement = imageWidth / ((x2 - x1) / increment)
        image = Image.new("RGB", (imageWidth, imageHeight), "white")
        draw = ImageDraw.Draw(image)
        for x in np.arange(x1, x2, increment): 
            for y in np.arange(y1, y2, increment):
                if(network.Classify([x,y]) == 0):
                    color = "red"
                else:
                    color = "blue"
                pixelCords = gui.coordinatesToPixels(x, y, xSize, ySize, imageWidth, imageHeight)
                draw.rectangle([(pixelCords[0], pixelCords[1]), (pixelCords[0] + pixelIncrement, pixelCords[1] + pixelIncrement)], fill=color)

        for datapoint in datapoints:
            pixelCords = gui.coordinatesToPixels(datapoint.inputs[0], datapoint.inputs[1], xSize, ySize, imageWidth, imageHeight)
            left = pixelCords[0] - radius
            top = pixelCords[1] - radius
            right = pixelCords[0] + radius
            bottom = pixelCords[1] + radius
            bbox = (left, top, right, bottom)
            if(datapoint.expectedOutput[0] > 0.99):
                draw.ellipse(bbox, fill="black")
            else:
                draw.ellipse(bbox, fill="white")

        # Convert the modified image to Tkinter-compatible format
        tk_image = ImageTk.PhotoImage(image)
        # Update the label with the modified image
        label.config(image=tk_image)
        label.image = tk_image  # Keep a reference to prevent garbage collection

# Create a window
window = tk.Tk()
window.title("Neural Network")
window.geometry("1000x800")

# Create a new image
imageWidth, imageHeight = 750, 750
image = Image.new("RGB", (imageWidth, imageHeight), "white")
tk_image = ImageTk.PhotoImage(image)


# Create a button widget
button = tk.Button(window, text="Draw Ellipse",  command=lambda: draw_network(0,0,100,100, datapoints))
button.grid(row = 0, column=1)

# Create a Tkinter label widget to display the image
label = tk.Label(window, image=tk_image)
label.grid(row = 1, column=1, rowspan=16)

# Start the GUI event loop
window.mainloop()