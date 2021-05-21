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
            nn.Linear(28*28, 256),
            nn.Dropout(p=0.6, inplace=True),
            nn.Sigmoid(),
            nn.Linear(256, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.flatten = nn.Flatten()
        
        self.Conv_Layers = nn.Sequential(
            nn.Conv2d(1, 3, 3),
            nn.ReLU(),
            nn.Conv2d(3, 12, 3),
            #nn.Dropout(p=0.6)
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.Linear_Layers = nn.Sequential(
            nn.Linear(1728, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.Conv_Layers(x)
        x = self.flatten(x)
        logits = self.Linear_Layers(x)
        return logits
    
    
# Initializing NN model
model = NeuralNetwork()
model = torch.load('./model.pth')
# print(list(model.parameters()))

# Initializing CNN model
conv_model = NeuralNetwork()
conv_model = torch.load('./conv_model.pth')
# print(list(model.parameters()))


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

# Initializing text
FNN_Guess_Text = '...'
CNN_Guess_Text = '...'

# Pretty clear what this does
def clear_canvas():
    t.clear()


def submit():
    '''
    This function does a bit:
        1) saves the canvas as a postscript
        2) Converts it to a PIL image object
        3) Resizes it and converts it to a 28x28x3 numpy array
        4) Converts np to grey scale and then a tensor 
        5) gets the model output and updates the label
    '''
    
    global FNN_Guess_Text
    global CNN_Guess_Text

    # Saving the canvas as a Postscript
    ps = t.getscreen().getcanvas().postscript(colormode = 'gray')
    # Converting postscript to image.postscript object
    img = Image.open(io.BytesIO(ps.encode('utf-8')))
    # converting image to Image.image object
    img = img.convert()
    # resizing image to be the appropriate input size for NN
    img = img.resize((28,28), Image.ANTIALIAS)
    # Converting the image to an array
    pix = np.array(img)
    
    
    # Grayscale = R / 3 + G / 3 + B / 3.
    # Grayscale  = 0.299R + 0.587G + 0.114B
    
    # here we are converting the image to be grayscale
    
    R = pix[:,:,0]
    G = pix[:,:,1]
    B = pix[:,:,2]
    
    # Createss an empty np array
    greyscale_img = np.zeros((pix.shape[0],pix.shape[1]))
    
    # These loops average each RGB pixel value and then saves it to the empty array
    for i in range(pix.shape[0]):
        for j in range(pix.shape[1]):
            greyscale_img[i][j] = (R[i][j]/3)+(B[i][j]/3)+(G[i][j]/3)
            greyscale_img[i][j] = 255 - greyscale_img[i][j]
    
    
    
    # plt.imshow(greyscale_img)
    # plt.show()
    
    # greyscale_img = greyscale_img.reshape((1,784))
    
    grey_tensor = torch.from_numpy(greyscale_img)
    
    #Formating input for FNN
    grey_tensor_lin = torch.flatten(grey_tensor)
    grey_tensor_lin = torch.reshape(grey_tensor_lin,(1,784))
    
    
    grey_tensor_2d = torch.reshape(grey_tensor, (1,1,28,28))
    
    # We dont want to update the NN, just make a prediction
    with torch.no_grad():
        #making a prediction
        pred = model(grey_tensor_lin.float())
        conv_pred = conv_model(grey_tensor_2d.float())
    
    # argmax retrieves the NN Guess in a tensor, .item returns an int/float
    FNN_Guess_Text = pred.argmax().item()
    #Updates the label in the GUI
    FNN_Guess.config(text = FNN_Guess_Text,font =("Courier", 28))
    
    # argmax retrieves the NN Guess in a tensor, .item returns an int/float
    CNN_Guess_Text = conv_pred.argmax().item()
    #Updates the label in the GUI
    CNN_Guess.config(text = CNN_Guess_Text,font =("Courier", 28))
    print(f'This is FNN: {pred}')
    print(f'This is CNN: {conv_pred}')
    


    


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
t.width(width=40) #40

t.ondrag(dragging)
screen.onscreenclick(set_mouse_pos)


button_clear = tk.Button(root, text = "Clear Canvas",  highlightbackground='gray', command=clear_canvas)
button_clear.place(relx=.75, rely=.8)

button_save = tk.Button(root, text ="Submit",  highlightbackground='gray', command=submit)
button_save.place(relx=.785, rely=.6)


FNN_Guess = tk.Label(root,text=FNN_Guess_Text,height = 1,width=5,bg='gray',highlightthickness=0,font =("Courier", 28))
FNN_Guess.place(relx=.75, rely=.2)

FNN_Label = tk.Label(root,text='Neural Net Guess:',height = 1,width=17,bg='gray',highlightthickness=0,font =("Courier", 16))
FNN_Label.place(relx=.69, rely=.1)

CNN_Guess = tk.Label(root,text=CNN_Guess_Text,height = 1,width=5,bg='gray',highlightthickness=0,font =("Courier", 28))
CNN_Guess.place(relx=.75, rely=.4)

CNN_Label = tk.Label(root,text='C Neural Net Guess:',height = 1,width=17,bg='gray',highlightthickness=0,font =("Courier", 16))
CNN_Label.place(relx=.69, rely=.3)

root.mainloop()

#python digit_recongnizer.py
