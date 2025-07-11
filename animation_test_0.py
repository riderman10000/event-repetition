from PIL import Image
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, show
import matplotlib.animation as animation
#this is crucial to animation in matplotlib
# %matplotlib notebook
#function to generate squares full of different colours
def next_shape(im, num_squares, sq_size):
    pix = im.load()
    #initialise plot location
    startx, starty = 0, 0
    for i in range(num_squares):
        startx += sq_size
        for j in range(num_squares):
            starty += sq_size
            rshade = np.random.randint(0, 256)
            gshade = np.random.randint(0, 256)
            bshade = np.random.randint(0, 256)
            for x in range(sq_size):
                        for y in range(sq_size):
                            value = (rshade, gshade, bshade)
                            pix[(startx + x) % im_size, (starty + y) % im_size] = value
    #return list of pixel tuples
    return list(im.getdata())
# Select the size (px) of each square + number of squares 
sq_size = 20
num_squares = 5
#create a figure to place the animation
fig = plt.figure()
#create a placeholder image inside the figure
im_size = sq_size * num_squares
im = Image.new('RGB', (im_size, im_size))
#create a list to store all the images in the format of a list of RGB pixels
im_list_pix = []
#generate a bunch of images in the form of a list of RGB pixels
for pic in range(10):
    im_list_pix.append(next_shape(im, num_squares, sq_size))
    
#create a list to store images converted from RGB pixel tuples to image format
img_array = []
#convert list of pixel tuples back to image
for i, v in enumerate(im_list_pix):
    im = Image.new('RGB', (100, 100))
    #put the pixel data into the image container
    im.putdata(im_list_pix[i])
    
    im = plt.imshow(im)
    img_array.append([im])  
#animate the results
#interval is the pause between images    
ani = animation.ArtistAnimation(fig, img_array, interval=500)
#display the output of the animation
plt.show()