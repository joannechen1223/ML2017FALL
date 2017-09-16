'''
ML_HW0_Q2
Date:2017.09.16
Author:b04701232 joannechen1223
'''
#!/usr/bin/env python3
import sys
from PIL import Image
img = Image.open(sys.argv[1])
pixel_data = img.load()
for i in range(img.size[0]):
	for j in range(img.size[1]):
		pixel_data[i,j] = (pixel_data[i,j][0]//2,pixel_data[i,j][1]//2,pixel_data[i,j][2]//2)
img.save("Q2.jpg")