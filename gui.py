from matplotlib import pyplot as plt
# from scipy import ndimage
import cv2
import numpy as np
image=cv2.imread("sample.jpg")
cv2.namedWindow("Signature")
cv2.imshow("Signature",image)
cv2.waitKey(0)
cv2.createTrackbar("zoom", "Signature", 0, 100, callback)