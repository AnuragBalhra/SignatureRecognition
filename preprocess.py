from __future__ import division
###########################################################################################################################
###########################################################################################################################
# from matplotlib import pyplot as plt
#from preprocess import *
# from scipy import ndimage

import cv2
import numpy as np

from os import listdir
def preprocess(img):
    # Convert to greyscale
    img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#     cv2.imshow('GrayScale Signature',img2gray)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    # Gaussian Blur
    # blur_img = cv2.GaussianBlur(img2gray,(5,5),0)
    blur_img = cv2.GaussianBlur(img2gray,(15,15),0)

#     cv2.imshow('Gausian Blur Signature',blur_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    # Dilation
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(blur_img,kernel,iterations = 3)
    # dilation = cv2.dilate(blur_img,kernel,iterations = 1)

#     cv2.imshow('Dilated Signature',dilation)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    # Thresholding
    ret, mask = cv2.threshold(dilation, 200, 255, cv2.THRESH_BINARY_INV)
    # mask_inv = cv2.bitwise_not(mask)

    # cv2.imshow('Preprocessed Signature',mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return mask
