import tkinter as tk

canvasWidth = 500
canvasHeight = 500

def handle_mouse_click(event):
    x = event.x
    y = event.y
    if event.num == 1:  # Left mouse button clicked
        with open("dataset1.txt", "a") as file:
            file.write("0,1,{},{}\n".format(x / canvasWidth, 1.0 - y / canvasHeight))
        canvas.create_oval(x-3, y-3, x+3, y+3, fill="blue")
    elif event.num == 3:  # Right mouse button clicked
        with open("dataset1.txt", "a") as file:
            file.write("1,0,{},{}\n".format(x / canvasWidth, 1.0-  y / canvasHeight))
        canvas.create_oval(x-3, y-3, x+3, y+3, fill="red")

root = tk.Tk()

canvas = tk.Canvas(root, width=canvasWidth, height=canvasHeight)
canvas.pack()

canvas.bind("<Button-1>", handle_mouse_click)  # Bind left mouse button click event
canvas.bind("<Button-3>", handle_mouse_click)  # Bind right mouse button click event

root.mainloop()
