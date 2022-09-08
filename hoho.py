from PIL import Image
  
# Opening the primary image (used in background)
img1 = Image.open(r"BACKGROUND_IMAGE_PATH")
  
# Opening the secondary image (overlay image)
img2 = Image.open(r"OVERLAY_IMAGE_PATH")
  
# Pasting img2 image on top of img1 
# starting at coordinates (0, 0)
img1.paste(img2, (0,0), mask = img2)
  
# Displaying the image
img1.show()
