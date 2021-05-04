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
# Setting the size and background color of the canvas
canvas = tk.Canvas(root,width =400, height =400, bg = 'white')
# Placing the canvas in the top left corner
canvas.place(relx=0,rely=0)

def clear_canvas():
    t.clear()

def save():

    # Saving the canvas as a Postscript
    ps = t.getscreen().getcanvas().postscript(colormode = 'gray')
    # Converting postscript to image.postscript object
    img = Image.open(io.BytesIO(ps.encode('utf-8')))
    #converting image to Image.image object
    img = img.convert()
    #resizing image to be the appropriate input size for NN
    img = img.resize((28,28), Image.ANTIALIAS)
    #Converting the image to an array
    pix = np.array(img)
    
    
    #Grayscale = R / 3 + G / 3 + B / 3.
    #Grayscale  = 0.299R + 0.587G + 0.114B
    
    #here we are converting the image to be grayscale
    
    R = pix[:,:,0]
    G = pix[:,:,1]
    B = pix[:,:,2]
    
    greyscale_img = np.zeros((pix.shape[0],pix.shape[1]))
    
    for i in range(pix.shape[0]):
        for j in range(pix.shape[1]):
            greyscale_img[i][j] = (R[i][j]/3)+(B[i][j]/3)+(G[i][j]/3)
            greyscale_img[i][j] = 255 - greyscale_img[i][j]

    
    #imgplot = plt.imshow(greyscale_img, cmap='gray')
    #plt.show()
    
    print('hi')

    


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
t.width(width=40)

t.ondrag(dragging)
screen.onscreenclick(set_mouse_pos)


button_clear = tk.Button(root, text = "Clear Canvas",  highlightbackground='gray', command=clear_canvas)
button_clear.place(relx=.75, rely=.8)

button_save = tk.Button(root, text ="Save",  highlightbackground='gray', command=save)
button_save.place(relx=.75, rely=.6)

root.mainloop()

#python digit_recongnizer.py