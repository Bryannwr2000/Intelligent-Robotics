import numpy as np
import cv2
import os.path
import random
from random import randint
import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image, ImageEnhance

def nums(first_number, last_number, step=1):
    return range(first_number, last_number+1, step)

iter = 0
    
#Fruits (Apple: 1, green_apple: 2, orange: 3, mango: 4, capsicum: 5)
for i in nums(1,5):
    for a in nums(0,2):
    #Loop for 5 different fruits and 3 fruit sample each
        for j in nums(0,14):
            #Loop for 15 different backgrounds
            for k in nums(1,80): #Number of picture
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
                    height = random.randint(20,160) #depending on the size u want
                elif a == 1:
                    height = random.randint(20,80) #depending on the size u want
                        
                width = height #assume we want square (chg ratio here later if needed)
                fruit = fruit.resize((4*width,3*height), Image.Resampling.LANCZOS) #resize image ratio 4:3
                
                #Random Brightness
                factor = round(random.uniform(0.65,1.30), 2) #0.65-1.3
                fruit = ImageEnhance.Brightness(fruit).enhance(factor)
                
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

#Loop for 2 fruit in one background
for i in nums(1,5): #Loop for 5 different main fruits
    if i == 1: index = [2,3,4,5]
    elif i == 2: index = [1,3,4,5]
    elif i == 3: index = [1,2,4,5]
    elif i == 4: index = [1,2,3,5]
    elif i == 5: index = [1,2,3,4]
        
    for b in index: #Loop for alternative fruits for matching
        for a in nums(0,2): #Loop for 5 different fruits and 2 fruit sample each
            for j in nums(0,14): #Loop for 10 different backgrounds
                for k in nums (1,30): #Loop for number of images
                    #Fruit 1 Image Path (Main Fruit)
                    filename = "sample/fruit/fruit_{}-{}.png" .format(i,a)
                    fruit1 = Image.open(filename)
                    #Fruit 2 Image Path (Alternate Fruit)
                    filename2 = "sample/fruit/fruit_{}-{}.png" .format(b,a)
                    fruit2 = Image.open(filename2)
                    # Background Image Path
                    filename3 = "sample/background/img_{}.png" .format(j)
                    background = Image.open(filename3)
                    
                    #Random Size
                    height = random.randint(20,80) #depending on the size u want
                    width = height #assume we want square (chg ratio here later if needed)
                    fruit1 = fruit1.resize((4*width,3*height), Image.Resampling.LANCZOS) #resize image ratio 4:3
                    
                    height = random.randint(20,80) #depending on the size u want
                    width = height #assume we want square (chg ratio here later if needed)
                    fruit2 = fruit2.resize((4*width,3*height), Image.Resampling.LANCZOS) #resize image ratio 4:3
                    
                    #Random Brightness
                    factor = round(random.uniform(0.65,1.30), 2) #0.65-1.1
                    fruit1 = ImageEnhance.Brightness(fruit1).enhance(factor)
                    
                    factor = round(random.uniform(0.65,1.30), 2) #0.65-1.1
                    fruit2 = ImageEnhance.Brightness(fruit2).enhance(factor)
                    
                    #Paste in fixed Position in Background (W = 640 x H = 480)
                    background.paste(fruit1, (0, 80),fruit1) #paste onto background image
                    background.paste(fruit2, (320, 80),fruit2) #paste onto background image
                    
                    #Convert to 3 channel (Remove RGBA)
                    impose = background.convert('RGB')
                    
                    #Save Superimposed Image
                    impose.save('network/scripts/dataset/images/image_{}.png'.format(iter))
                    
                    #Data Labelling
                    fruit1 = fruit1.convert('L') #Convert image to greyscale
                    fruit_array1 = np.array(fruit1) #Convert image to array
                    fruit_array1[fruit_array1 > 0] = 255 #Assign pixel values
                    fruit_label1 = Image.fromarray(fruit_array1) #Convert array to image
                    
                    fruit2 = fruit2.convert('L') #Convert image to greyscale
                    fruit_array2 = np.array(fruit2) #Convert image to array
                    fruit_array2[fruit_array2 > 0] = 254 #Assign pixel values
                    fruit_label2 = Image.fromarray(fruit_array2) #Convert array to image
                
                    #Make a black background and superimpose 2nd time
                    bg_array = np.zeros((480, 640))
                    bg = Image.fromarray(bg_array, 'L')
                    bg.paste(fruit_label1, (0, 80),fruit_label1) #paste fruit1 onto background image
                    bg.paste(fruit_label2, (320, 80),fruit_label2) #paste fruit2 onto background image
                    
                    #Convert everything to Label
                    label_array = np.array(bg) #background array
                    label_array[label_array==255] = i #Assign pixel values to i
                    label_array[(label_array<255)&(label_array>10)] = b #Assign pixel values to b
                    label = Image.fromarray(label_array)
                    
                    #Save Labelled Image
                    label.save('network/scripts/dataset/labels/image_{}_label.png'.format(iter))
                    
                    iter +=1