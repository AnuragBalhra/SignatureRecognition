# libraries to include 
from matplotlib import pyplot as plt
# from scipy import ndimage
import cv2
import numpy as np

# take image input
print("cv2.__version__",str(cv2.__version__) )

# Load signature image
img = cv2.imread("1.jpeg")
h,w=img.shape[:2]
img = img[270:h, 1600:w-100]
original_img = img
img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 150, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# finding contours 
img = mask_inv

# for i in range(3):
img = cv2.medianBlur(img,3)
img = cv2.GaussianBlur(img,(15,15),0)

cv2.imshow('signature',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((5,1),np.uint8)
print(kernel)
# erosion = img
erosion = cv2.erode(img,kernel,iterations = 3)

kernel = np.ones((1,5),np.uint8)
dilation = cv2.dilate(erosion,kernel,iterations = 3)
# dilation = erosion
# cv2.imshow('blurred image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('erosion',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('dilation',dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

# img = dilation

cv2.waitKey(0)

cv2.destroyAllWindows()

# ret,thresh = cv2.threshold(img2gray,127,255,0)
_, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


cv2.imshow("show contours", original_img)
# cv2.waitKey(0)

# cv2.waitKey(0)
signs = []
inc = 20

start = 270
end = 270 + inc
while(end<h):
    if(end>=h):
        print("out of bound indexes ")
        break
    cv2.imshow("show contours", original_img[start:end,:]) 
    but = cv2.waitKey(0)
    if(but==ord('c')):
        signs.append({start,end})
        start = end
        end =end + inc
    elif(but == ord('w')):
        start = start - inc
    elif(but == ord('d')):
        end = end + inc
    elif(but==ord('u')):
    	end=end-inc
    cv2.destroyAllWindows()

    
signs.append({start,end})

print(signs)