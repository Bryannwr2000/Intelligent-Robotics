import numpy as np
import cv2
import os.path
import random
from random import randint
import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image

def nums(first_number, last_number, step=1):
    return range(first_number, last_number+1, step)

iter = 0
    
#Fruits (Apple: 1, green_apple: 2, orange: 3, mango: 4, capsicum: 5)
for i in nums(1,5):
    for a in nums(0,1):
    #Loop for 5 different fruits and 2 fruit sample each
        for j in nums(0,14):
            #Loop for 15 different backgrounds
            for k in nums(1,50): #Number of picture
                #Fruit Image Path
                filename = "sample/fruit/fruit_{}-{}.png" .format(i,a)
                fruit = Image.open(filename)
                # Background Image Path
                filename2 = "sample/background/img_{}.png" .format(j)
                background = Image.open(filename2)
                
                #Random Orientation
                angle = random.randint(0,360)
                fruit = fruit.rotate(angle)
                
                #Random Size
                if a == 0:
                    height = random.randint(50,160) #depending on the size u want
                elif a == 1:
                    height = random.randint(40,80) #depending on the size u want
                        
                width = height #assume we want square (chg ratio here later if needed)
                fruit = fruit.resize((4*width,3*height), Image.Resampling.LANCZOS) #resize image ratio 4:3
                
                #Paste in random Position in Background
                #640x480 WxH
                if height > 120 or (a == 1 and height > 60):
                    img_posx = random.randint(-50,50) #640 width
                    img_posy = random.randint(-50,50) #480 height
                else:
                    img_posx = random.randint(110,430) #640 width
                    img_posy = random.randint(130,310) #480 height
                       
                background.paste(fruit, (img_posx, img_posy),fruit) #paste onto background image
                
                #Convert to 3 channel (Remove RGBA)
                impose = background.convert('RGB')
                
                #Save Superimposed Image
                impose.save('network/scripts/dataset/images/image_{}.png'.format(iter))
                
                #Data Labelling
                fruit = fruit.convert('L') #Convert image to greyscale
                fruit_array = np.array(fruit) #Convert image to array
                fruit_array[fruit_array > 0] = 255 #Assign pixel values
                fruit_label = Image.fromarray(fruit_array) #Convert array to image
                
                #Make a black background and superimpose 2nd time
                bg_array = np.zeros((480, 640))
                bg = Image.fromarray(bg_array, 'L')
                bg.paste(fruit_label, (img_posx, img_posy),fruit_label) #paste onto background image
                
                #Convert everything to Label
                label_array = np.array(bg)
                label_array[label_array==255] = i #Assign pixel values
                label = Image.fromarray(label_array)
                
                #Save Labelled Image
                label.save('network/scripts/dataset/labels/image_{}_label.png'.format(iter))
                
                iter +=1