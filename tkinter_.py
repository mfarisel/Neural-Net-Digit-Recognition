#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:40:12 2021

@author: mattfariselli
"""
from tkinter import Tk, Label, Button

import turtle
import tkinter as tk

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

button_clear = tk.Button(root, text ="Clear Canvas",  highlightbackground='gray',command=clear_canvas)
button_clear.place(relx=.75, rely=.8)


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

t.pencolor("#ff0000") # Red
t.speed(speed=0)
t.width(width=10)

t.ondrag(dragging)
screen.onscreenclick(set_mouse_pos)


root.mainloop()

#python tkinter_.py