import NeuralNetwork as NN
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image, ImageDraw

network = NN.Network([2,3,2])

divider = 1.0

startX = 0
startY = 0
endX = 100
endY = 100

def on_slider_move1(value):
    network.layers[0].weights[0][0] = float(value)
    draw_network(startX, startY, endX, endY)

def on_slider_move2(value):
    network.layers[0].weights[0][1] = float(value)
    draw_network(startX, startY, endX, endY)

def on_slider_move3(value):
    network.layers[0].weights[0][2] = float(value)
    draw_network(startX, startY, endX, endY)

def on_slider_move4(value):
    network.layers[0].weights[1][0] = float(value)
    draw_network(startX, startY, endX, endY)

def on_slider_move5(value):
    network.layers[0].weights[1][1] = float(value)
    draw_network(startX, startY, endX, endY)

def on_slider_move6(value):
    network.layers[0].weights[1][2] = float(value)
    draw_network(startX, startY, endX, endY)

def on_slider_move7(value):
    network.layers[1].weights[0][0] = float(value)
    draw_network(startX, startY, endX, endY)

def on_slider_move8(value):
    network.layers[1].weights[0][1] = float(value)
    draw_network(startX, startY, endX, endY)

def on_slider_move9(value):
    network.layers[1].weights[1][0] = float(value)
    draw_network(startX, startY, endX, endY)

def on_slider_move10(value):
    network.layers[1].weights[1][1] = float(value)
    draw_network(startX, startY, endX, endY)

def on_slider_move11(value):
    network.layers[1].weights[2][0] = float(value)
    draw_network(startX, startY, endX, endY)

def on_slider_move12(value):
    network.layers[1].weights[2][1] = float(value)
    draw_network(startX, startY, endX, endY)

def on_slider_move13(value):
    network.layers[0].bias[0] = float(value)
    draw_network(startX, startY, endX, endY)

def on_slider_move14(value):
    network.layers[0].bias[1] = float(value)
    draw_network(startX, startY, endX, endY)

def on_slider_move15(value):
    network.layers[0].bias[2] = float(value)
    draw_network(startX, startY, endX, endY)

def on_slider_move16(value):
    network.layers[1].bias[0] = float(value)
    draw_network(startX, startY, endX, endY)

def on_slider_move17(value):
    network.layers[1].bias[1] = float(value)
    draw_network(startX, startY, endX, endY)


def draw_network(x1, y1, x2, y2):
    increment = 2
    pixelIncrement = width / ((x2 - x1) / increment)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    # Modify the image (draw an ellipse)
    for x in np.arange(x1, x2, increment): 
        for y in np.arange(y1, y2, increment):
            if(network.Classify([x,y]) == 0):
                color = "red"
            else:
                color = "blue"
            X = (x / (x2 - x1)) * width
            Y = (y / (y2 - y1)) * height
            draw.rectangle([(X, Y), (X+pixelIncrement, Y+pixelIncrement)], fill=color)
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
width, height = 750, 750
image = Image.new("RGB", (width, height), "white")
tk_image = ImageTk.PhotoImage(image)


slider1 = tk.Scale(window, from_=-25, to=25, orient=tk.HORIZONTAL,  command=on_slider_move1)
slider1.grid(row = 0, column=0)

slider2 = tk.Scale(window, from_=-25, to=25, orient=tk.HORIZONTAL,  command=on_slider_move2)
slider2.grid(row = 1, column=0)

slider3 = tk.Scale(window, from_=-25, to=25, orient=tk.HORIZONTAL,  command=on_slider_move3)
slider3.grid(row = 2, column=0)

slider4 = tk.Scale(window, from_=-25, to=25, orient=tk.HORIZONTAL,  command=on_slider_move4)
slider4.grid(row = 3, column=0)

slider5 = tk.Scale(window, from_=-25, to=25, orient=tk.HORIZONTAL,  command=on_slider_move5)
slider5.grid(row = 4, column=0)

slider6 = tk.Scale(window, from_=-25, to=25, orient=tk.HORIZONTAL,  command=on_slider_move6)
slider6.grid(row = 5, column=0)

slider7 = tk.Scale(window, from_=-25, to=25, orient=tk.HORIZONTAL,  command=on_slider_move7)
slider7.grid(row = 6, column=0)

slider8 = tk.Scale(window, from_=-25, to=25, orient=tk.HORIZONTAL,  command=on_slider_move8)
slider8.grid(row = 7, column=0)

slider9 = tk.Scale(window, from_=-25, to=25, orient=tk.HORIZONTAL,  command=on_slider_move9)
slider9.grid(row = 8, column=0)

slider10 = tk.Scale(window, from_=-25, to=25, orient=tk.HORIZONTAL,  command=on_slider_move10)
slider10.grid(row = 9, column=0)

slider11 = tk.Scale(window, from_=-25, to=25, orient=tk.HORIZONTAL,  command=on_slider_move11)
slider11.grid(row = 10, column=0)

slider12 = tk.Scale(window, from_=-25, to=25, orient=tk.HORIZONTAL,  command=on_slider_move12)
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
button = tk.Button(window, text="Draw Ellipse",  command=draw_network)
button.grid(row = 0, column=1)

# Create a Tkinter label widget to display the image
label = tk.Label(window, image=tk_image)
label.grid(row = 1, column=1, rowspan=16)

# Start the GUI event loop
window.mainloop()