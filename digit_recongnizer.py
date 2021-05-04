import turtle
import tkinter as tk
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
import time


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

# Initializing NN model
model = NeuralNetwork()
model = torch.load('./model.pth')


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

#initializing text
NN_Guess_Text = '...'

def clear_canvas():
    t.clear()

def submit():
    global NN_Guess_Text

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
    
    greyscale_img = greyscale_img.reshape((1,784))
    
    
    grey_tens = torch.from_numpy(greyscale_img)
    
    with torch.no_grad():
        pred = model(grey_tens.float())
    
    NN_Guess_Text = pred.argmax().item()
    NN_Guess.config(text = NN_Guess_Text,font =("Courier", 28))

    

    


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

button_save = tk.Button(root, text ="Submit",  highlightbackground='gray', command=submit)
button_save.place(relx=.785, rely=.6)


NN_Guess = tk.Label(root,text=NN_Guess_Text,height = 1,width=5,bg='gray',highlightthickness=0,font =("Courier", 28))
NN_Guess.place(relx=.75, rely=.3)

NN_Label = tk.Label(root,text='Neural Net Guess:',height = 1,width=17,bg='gray',highlightthickness=0,font =("Courier", 16))
NN_Label.place(relx=.69, rely=.2)


root.mainloop()

#python digit_recongnizer.py
