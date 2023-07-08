import NeuralNetwork as NN
import gui
import tkinter as tk
from PIL import ImageTk, Image, ImageDraw
import threading
import time

network = NN.Network([2,3,3,2])

Size = 1.0        # Size of coordinate plot
PixelSize = 750   # Pixel size of image
isLearning = False

# Open the file
with open('dataset1.txt', 'r') as file:
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

def draw_network():
    # Convert the modified image to Tkinter-compatible format
    tk_image = gui.getNetworkImage(network, Size, 0.0025, PixelSize, datapoints)
    # Update the label with the modified image
    lblImage.config(image=tk_image)
    lblImage.image = tk_image  # Keep a reference to prevent garbage collection

def save_network():
    network.write_weights_bias_to_file("NeuralNetwork.txt")
    
def network_learn():
    while isLearning == True:
        print("learning", network.AverageCost(datapoints))
        for i in range(100):
            network.Learn(datapoints, 0.3)

def update_screen():
    while 1:
        draw_network()
        time.sleep(2.5)
    
def StartLearn():
    global isLearning
    isLearning = True
    threading.Thread(target=network_learn).start()

def StopLearn():
    global isLearning
    isLearning = False

window = tk.Tk()
window.title("Neural Network")
window.geometry("1200x900")

image = Image.new("RGB", (PixelSize, PixelSize), "white")
tk_image = ImageTk.PhotoImage(image)

btnStartLearn = tk.Button(window, text="Start Learning",  command=StartLearn)
btnStartLearn.grid(row = 0, column=1)

btnStopLearn = tk.Button(window, text="Stop Learning",  command=StopLearn)
btnStopLearn.grid(row = 0, column=2)

btnDrawNetwork = tk.Button(window, text="Draw network",  command=draw_network)
btnDrawNetwork.grid(row = 0, column=3)

btnSaveNetwork = tk.Button(window, text="Save network",  command=save_network)
btnSaveNetwork.grid(row = 0, column=4)

lblImage = tk.Label(window, image=tk_image)
lblImage.grid(row = 1, column=1, rowspan=16, columnspan=6)

# Start the GUI event loop
window.mainloop()