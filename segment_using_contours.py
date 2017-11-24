# '''

from preprocess import *
# Extract Signatures using contour method
def findContours(img_name):
    # Load signature image
    img = cv2.imread("Signatures By Names/Vijay/"+img_name+".jpg")
    # img = img[:,1560:]
    # Convert to greyscale
    img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#     cv2.imshow('GrayScale Signature',img2gray[:,1450:])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     plt.imshow(img2gray, cmap = 'gray', interpolation='bicubic')
#     plt.show()
    
    # Gaussian Blur
    blur_img = cv2.GaussianBlur(img2gray,(15,15),0)

#     cv2.imshow('Gausian Blur Signature',blur_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    # Erosion in y direction
    kernel = np.ones((5,1),np.uint8)
    erosion = cv2.erode(blur_img,kernel,iterations = 3)
    
    # Dilation in x direction
    kernel = np.ones((1,5),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 3)

#     cv2.imshow('Dilated Signature',dilation)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    # Thresholding
    ret, mask = cv2.threshold(dilation, 200, 255, cv2.THRESH_BINARY_INV)
    # mask_inv = cv2.bitwise_not(mask)

#     cv2.imshow('Thresholded Signature',mask)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


    _, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    img = preprocess(img)
    # cv2.namedWindow("contours", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
    # cv2.drawContours(img, contours, -1, (0,255,0),3)
    X = 1700
    w = 800
    Y = 0
    h = 2500

    # cv2.rectangle(img,(X,Y),(X+w,Y+h),(0,0,250),8)
        
    for c in contours:
        rect = cv2.boundingRect(c)
        # if rect[2] < 100 or rect[3] < 100: continue
        # print cv2.contourArea(c)
        x,y,w,h = rect
        # if(x<X ):
            # continue
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        # cv2.putText(img,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0)
    # ims = cv2.resize(img,(2532//3,3502//3))
    cv2.imshow("contours",img)
    cv2.waitKey(0)
    # print(contours)
    # idx=0
    # cutoff = 10000
    # for cnt in contours:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     # if(w*h < cutoff):
    #         # continue
    #     idx += 1
    #     roi = img[y:y+h,x:x+w]
    #     # cv2.imwrite(str(idx) + '.jpg', roi)
    #     # cv2.imwrite("contour_extracted_images/"+img_name+"_"+str(idx)+".jpg", roi)

    #     cv2.imshow(img_name+'_'+str(idx), roi)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
#         plt.imshow(roi, cmap = 'gray', interpolation='bicubic')
#         plt.show()


from os import listdir
import cv2
import numpy as np
mypath = "Signatures By Names/Vijay/"
onlyfiles = [f.split('.')[0] for f in listdir(mypath)]
print onlyfiles

findContours(onlyfiles[2])
# for x in onlyfiles:
#     findContours(x)
    
    
# '''
    