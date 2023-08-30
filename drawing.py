import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import NeuralNetwork as NN
import Matrix

network = NN.Network([784,200,10])
network.ReadNetworkFromFile('digitsNetwork.txt')
drawing = Image.new("L", (200, 200), color=0)
draw = ImageDraw.Draw(drawing)

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=20)
    draw.line([x1, y1, x2, y2], fill=255, width=20)

def buttonRelease(event):
    convert_to_greyscale()

def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 200, 200), fill=0)

def displayImage():
    greyscale_img = Image.new("L", (28, 28), color=0)
    greyscale_draw = ImageDraw.Draw(greyscale_img)
    scaled_drawing = drawing.resize((28, 28))
    Matrix.display_image(scaled_drawing)

def convert_to_greyscale():
    greyscale_img = Image.new("L", (28, 28), color=0)
    greyscale_draw = ImageDraw.Draw(greyscale_img)
    scaled_drawing = drawing.resize((28, 28))
    greyscale_img.paste(scaled_drawing, (0, 0))
    image_array = np.array(greyscale_img) / 255
    inputs = []
    for row in range(0, 28):
        for col in range(0,28):
            inputs.append( float(image_array[row,col]) )
    text_label.config(text=network.Classify(inputs))

def GetNetworkGuess():


root = tk.Tk()
root.title("Drawing Canvas to Greyscale Image")

canvas = tk.Canvas(root, bg="white", width=200, height=200)
canvas.pack()

canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", buttonRelease)
Display_Image = tk.Button(root, text="Display_Image", command=displayImage)


text_label = tk.Label(root, text="text")
clear_button = tk.Button(root, text="Clear Canvas", command=clear_canvas)
clear_button.pack()
text_label.pack()
Display_Image.pack()

root.mainloop()
