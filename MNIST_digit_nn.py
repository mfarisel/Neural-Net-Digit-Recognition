#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:16:45 2021

@author: mattfariselli
"""

import os
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np


file_name = os.path.abspath('six.JPG')
data = io.imread(file_name,as_gray=True)




aspect_ratio = data.shape[0]/data.shape[1]


'''
The common factors are:
1, 2, 3, 4, 6, 7, 8, 9, 12, 
14, 16, 18, 21, 24, 28, 36, 
42, 48, 56, 63, 72, 84, 112, 
126, 144, 168, 252, 336, 504, 1008
'''

# I think the best method may be to find the corners and then go from there
# -> Thinking back on it this may lead to some interference from random dark spots
# --> The best way to go about this will probably be some sort of recursive function
# ---> This way it will slowly widdle down the photo until it is mostly the number

step = 1008

height = int(data.shape[0]/step)
width = int(data.shape[1]/step)

dex = np.zeros((height,width))

for h in range(height):
    for w in range(width):
        dex[h][w] = data[h*step:(h+1)*step,w*step:(w+1)*step].min()


data2 = data[0*step:(0+1)*step,1*step:(1+1)*step]

step = 336
height = int(data2.shape[0]/step)
width = int(data2.shape[1]/step)

dex2 = np.zeros((height,width))
for h in range(height):
    for w in range(width):
        dex2[h][w] = data2[h*step:(h+1)*step,w*step:(w+1)*step].min()
    

plt.imshow(dex2, cmap='hot', interpolation='nearest')
plt.show()

data3 = data2[2*step:(2+1)*step,0*step:(0+1)*step]




'''
Useful for showing heat plot:
    
plt.imshow(dex, cmap='hot', interpolation='nearest')
plt.show()

Resizing tool:

image_resized = resize(data_splice, (28, 28),
                       anti_aliasing=True)


'''

