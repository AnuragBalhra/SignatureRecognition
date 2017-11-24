import cv2
import os

dirs=os.listdir("Signatures By Names/")
ofile = open('train_images2.txt','w')
for folder in dirs:
	#print inner

	path="Signatures By Names/"+folder+'/'
	images=os.listdir(path)
	for pic in images:
		img = cv2.imread(path+pic)
		ofile.write(pic.split('.')[0]+'\t'+folder+'\t1\n')
ofile.close()
