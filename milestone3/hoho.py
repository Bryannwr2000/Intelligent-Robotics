from PIL import Image

# Opening the primary image (used in background)
img1 = Image.open(r"Arena\img_0.png")
  
# Opening the secondary image (overlay image)
img2 = Image.open(r"Capsicum\capsicum\img_1.png")
img2_half = img2.resize((200, 200))  
# Pasting img2 image on top of img1 
# starting at coordinates (0, 0)
img1.paste(img2_half, (-50,100), mask = img2_half)
#(x,y) x: -ve (left), +ve (right)
#      y: -ve (up), +ve (down)
# Displaying the image
img1.show()
