from PIL import Image
import random
from random import randint
import matplotlib.pyplot as plt
from matplotlib import image

i = 1
a = 1
j = 1


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
    height = 20#random.randint(50,160) #depending on the size u want
elif a == 1:
    height = 20#random.randint(40,80) #depending on the size u want
        
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
background.show()