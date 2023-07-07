import NeuralNetwork as NN
import gui
import tkinter as tk
from PIL import ImageTk, Image, ImageDraw

network = NN.Network([2,2])
network.PrintNetwork()
#network.read_weights_bias_from_file("NeuralNetwork.txt")
network.PrintNetwork()

Size = 1.0        # Size of coordinate plot
PixelSize = 750   # Pixel size of image

weight0Divider = 50.0
weight1Divider = 10.0
bias0Divider = 10.0
bias1Divider = 20.0

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



def on_slider_move1(value):
    network.layers[0].weights[0][0] = float(value) / weight0Divider
    draw_network()

def on_slider_move2(value):
    network.layers[0].weights[0][1] = float(value) / weight0Divider
    draw_network()

def on_slider_move3(value):
    network.layers[0].weights[0][2] = float(value) / weight0Divider
    draw_network()

def on_slider_move4(value):
    network.layers[0].weights[1][0] = float(value) / weight0Divider
    draw_network()

def on_slider_move5(value):
    network.layers[0].weights[1][1] = float(value) / weight0Divider
    draw_network()

def on_slider_move6(value):
    network.layers[0].weights[1][2] = float(value) / weight0Divider
    draw_network()

def on_slider_move7(value):
    network.layers[1].weights[0][0] = float(value) / weight1Divider
    draw_network()

def on_slider_move8(value):
    network.layers[1].weights[0][1] = float(value) / weight1Divider
    draw_network()

def on_slider_move9(value):
    network.layers[1].weights[1][0] = float(value) / weight1Divider
    draw_network()

def on_slider_move10(value):
    network.layers[1].weights[1][1] = float(value) / weight1Divider
    draw_network()

def on_slider_move11(value):
    network.layers[1].weights[2][0] = float(value) / weight1Divider
    draw_network()

def on_slider_move12(value):
    network.layers[1].weights[2][1] = float(value) / weight1Divider
    draw_network()

def on_slider_move13(value):
    network.layers[0].bias[0] = float(value) / bias0Divider
    draw_network()

def on_slider_move14(value):
    network.layers[0].bias[1] = float(value) / bias0Divider
    draw_network()

def on_slider_move15(value):
    network.layers[0].bias[2] = float(value) / bias0Divider
    draw_network()

def on_slider_move16(value):
    network.layers[1].bias[0] = float(value) / bias1Divider
    draw_network()

def on_slider_move17(value):
    network.layers[1].bias[1] = float(value) / bias1Divider
    draw_network()

def draw_network():
    print(network.AverageCost(datapoints))
    # Convert the modified image to Tkinter-compatible format
    tk_image = gui.getNetworkImage(network, Size, 0.005, PixelSize, datapoints)
    # Update the label with the modified image
    label.config(image=tk_image)
    label.image = tk_image  # Keep a reference to prevent garbage collection

def save_network():
    network.write_weights_bias_to_file("NeuralNetwork.txt")
    
def network_learn():
    for i in range(100):
        network.Learn(datapoints, 0.02)

# Create a window
window = tk.Tk()
window.title("Neural Network")
window.geometry("1200x900")

image = Image.new("RGB", (PixelSize, PixelSize), "white")
tk_image = ImageTk.PhotoImage(image)

slider1 = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL,  command=on_slider_move1)
slider1.grid(row = 0, column=0)

slider2 = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL,  command=on_slider_move2)
slider2.grid(row = 1, column=0)

slider3 = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL,  command=on_slider_move3)
slider3.grid(row = 2, column=0)

slider4 = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL,  command=on_slider_move4)
slider4.grid(row = 3, column=0)

slider5 = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL,  command=on_slider_move5)
slider5.grid(row = 4, column=0)

slider6 = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL,  command=on_slider_move6)
slider6.grid(row = 5, column=0)

slider7 = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL,  command=on_slider_move7)
slider7.grid(row = 6, column=0)

slider8 = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL,  command=on_slider_move8)
slider8.grid(row = 7, column=0)

slider9 = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL,  command=on_slider_move9)
slider9.grid(row = 8, column=0)

slider10 = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL,  command=on_slider_move10)
slider10.grid(row = 9, column=0)

slider11 = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL,  command=on_slider_move11)
slider11.grid(row = 10, column=0)

slider12 = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL,  command=on_slider_move12)
slider12.grid(row = 11, column=0)

slider13 = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL,  command=on_slider_move13)
slider13.grid(row = 12, column=0)

slider14 = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL,  command=on_slider_move14)
slider14.grid(row = 13, column=0)

slider15 = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL,  command=on_slider_move15)
slider15.grid(row = 14, column=0)

slider16 = tk.Scale(window, from_=-100, to=100, orient=tk.HORIZONTAL,  command=on_slider_move16)
slider16.grid(row = 15, column=0)

# Create a button widget
button = tk.Button(window, text="Draw network",  command=draw_network)
button.grid(row = 0, column=1)

# Create a button widget
button = tk.Button(window, text="Save network",  command=save_network)
button.grid(row = 0, column=2)

# Create a button widget
button = tk.Button(window, text="Learn",  command=network_learn)
button.grid(row = 0, column=3)

# Create a Tkinter label widget to display the image
label = tk.Label(window, image=tk_image)
label.grid(row = 1, column=1, rowspan=16, columnspan=6)

draw_network()

# Start the GUI event loop
window.mainloop()