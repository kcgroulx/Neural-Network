import tkinter as tk
import NeuralNetwork as NN
import numpy as np
from PIL import ImageTk, Image, ImageDraw

def coordinatesToPixels(x:float, y:float, Size:float, PixelSize:int):
    ratio = PixelSize / Size
    return [(int)(x * ratio), (int)(PixelSize - (y * (ratio)))]

def getNetworkImage(network:NN.Network, Size:float , increment:float , PixelSize:int , datapoints:list[NN.Datapoint]):
    radius = 5
    pixelIncrement = PixelSize / (Size / increment)
    image = Image.new("RGB", (PixelSize, PixelSize), "white")
    draw = ImageDraw.Draw(image)
    x = 0.0
    #draws network classification
    while x < Size:
        y = 0.0
        while y < Size:
            color = "#BBDEFB"
            if(network.Classify([x,y]) == 0):
                color = "#FF9999"
            pixelCords = coordinatesToPixels(x, y, Size, PixelSize)
            draw.rectangle([(pixelCords[0], pixelCords[1]), (pixelCords[0] + pixelIncrement, pixelCords[1] + pixelIncrement)], fill=color)
            y += increment
        x += increment
    #draws datapoints
    if (datapoints != None):
        for datapoint in datapoints:
            pixelCords = coordinatesToPixels(datapoint.inputs[0], datapoint.inputs[1], Size, PixelSize)
            bbox = (pixelCords[0] - radius, pixelCords[1] - radius, pixelCords[0] + radius, pixelCords[1] + radius)
            if(datapoint.expectedOutput[0] > 0.99):
                draw.ellipse(bbox, fill="#990000")
            else:
                draw.ellipse(bbox, fill="#000099")
    return ImageTk.PhotoImage(image)

