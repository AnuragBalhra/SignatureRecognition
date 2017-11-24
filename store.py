# libraries to include 
from matplotlib import pyplot as plt
# from scipy import ndimage
import cv2
import numpy as np
import os

outer=os.listdir("Signatures By Names/")
for inner in outer:
	#print inner
	images=os.listdir("Signatures By Names/"+inner+'/')
	path="Signatures By Names/"+inner+'/'

	for pic in images:


		img = cv2.imread(path+pic)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		final = cv2.medianBlur(gray,5)
		h,w=final.shape[:2]
		rows=h
		cols=w
		ret,thresh = cv2.threshold(final,140,255,cv2.THRESH_BINARY_INV)
		#cv2.imshow('newimage',thresh)
		#cv2.waitKey(0)
		#print(thresh[10][10])

		top=(0,0)
		down=(0,0)
		left=(0,0)
		right=(0,0)
		for i in range(rows):
			for j in range(cols):
				if thresh[i][j]>=140:
					if top==(0,0):
						top=(i,j)
					down=(i,j)		
			

		for i in range(cols):
			for j in range(rows):
				if thresh[j][i]>=140:
					if left==(0,0):
						left=(j,i)
					right=(j,i)

		

		 gfinal_img = img[top[0]:down[0],left[1]:right[1]]
		 cv2.imwrite(path+pic,final_img)

