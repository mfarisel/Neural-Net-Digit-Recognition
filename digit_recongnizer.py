import turtle
import tkinter as tk
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



# Initiating the window
root = tk.Tk()
# making it so the window is not resizable
root.resizable(width=False, height=False)
# This sets the size of the window
root.geometry('600x400')
root.configure(bg='gray')

canvas = tk.Canvas(root,width =400, height =400, bg = 'white')
canvas.place(relx=0,rely=0)

def clear_canvas():
    t.clear()

def save():

    #img_data = io.BytesIO()
    ps = t.getscreen().getcanvas().postscript(colormode = 'gray')
    #Need to figure out how to conver postscript to numpy
    img = Image.open(io.BytesIO(ps.encode('utf-8')))
    #pix = np.asarray(img)
    #img2 = img.convert(mode='RGB')
    img2 = img.convert()
    
    pix = np.array(img2)
    imgplot = plt.imshow(pix)
    plt.show()

    
    print(pix.shape)
    


def dragging(x, y):  # These parameters will be the mouse position
    t.down()
    t.ondrag(None)
    t.setheading(t.towards(x, y))
    t.goto(x, y)
    t.ondrag(dragging)

def set_mouse_pos(x,y):
    t.up()
    t.goto(x,y)


screen = turtle.TurtleScreen(canvas)
t = turtle.RawTurtle(screen)

#t = turtle.RawTurtle(canvas)
#screen = turtle.TurtleScreen(canvas)

t.pencolor("black") 
t.speed(speed=0)
t.width(width=10)

t.ondrag(dragging)
screen.onscreenclick(set_mouse_pos)


button_clear = tk.Button(root, text = "Clear Canvas",  highlightbackground='gray', command=clear_canvas)
button_clear.place(relx=.75, rely=.8)

button_save = tk.Button(root, text ="Save",  highlightbackground='gray', command=save)
button_save.place(relx=.75, rely=.6)

root.mainloop()

#python tkinter2.py