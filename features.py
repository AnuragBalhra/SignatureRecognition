from __future__ import division
import cv2
from preprocess import *

def find_aspect_ratio(img):
    h,w = img.shape[:2]
    return h/w

def find_mass(img):
    count = 0
    h,w = img.shape[:2]
    for y in img:
        for x in y:
            if(x!=0):
                # print(x)
                count = count+1
    # print "mass found to be ",count/(h*w)
    return count/(h*w)

def find_closed_area(img):
    h,w = img.shape[:2]
    new_img = img.copy()
    for i in range(h):
        for j in range(w):
            if(img[i][j]==255):
#                 cv2.imshow('Thresholded Signature',new_img)
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()
                break
            else:
                new_img[i][j] = 100
    for i in range(h):
        for j in range(w):
            if(img[i][w-j-1]==255):
#                 cv2.imshow('Thresholded Signature',new_img)
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()
                break
            else:
                new_img[i][w-j-1] = 100
    for j in range(w):
        for i in range(h):
            if(img[i][j]==255):
#                 cv2.imshow('Thresholded Signature',new_img)
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()
                break
            else:
                new_img[i][j] = 100
    for j in range(w):
        for i in range(h):
            if(img[h-i-1][j]==255):
#                 cv2.imshow('Thresholded Signature',new_img)
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()
                break
            else:
                new_img[h-i-1][j] = 100
                
    count = 0
    for y in new_img:
        for x in y:
            if(x==100):
#                 print(x)
                count = count+1
#     cv2.imshow('Thresholded Signature',new_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    return count/(h*w)

def centroid(img):
    M = cv2.moments(img)
    
    if M["m00"] == 0:
        M["m00"] = 0.1
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx,cy)

def find_centroid_ratio(img):
    h,w = img.shape[:2]
    mask = img.copy()
    c1x,c1y = centroid(mask[:,:int(w/3)])
    c2x,c2y = centroid(mask[:,int(w/3):2*int(w/3)])
    c2x = c2x+int(w/3)
    c3x,c3y = centroid(mask[:,2*int(w/3):w-1])
    c3x = c3x+2*int(w/3)

    centroid_ratio = (c2x-c1x)/(c3x-c2x)

#     print(centroid(mask[:,:int(w/3)]))
#     print(centroid(mask[:,int(w/3):2*int(w/3)]))
#     print(centroid(mask[:,2*int(w/3):w-1]))
#     print(centroid_ratio)
    return centroid_ratio

def center_of_mass(img):
    sumx = 0
    sumy = 0
    count = 0
    h,w = img.shape[:2]
    for y in range(h):
        for x in range(w):
            if(img[y][x]==255):
                sumx = sumx + x
                sumy = sumy + y
                count = count + 1
    
    if(count == 0):
        return (0,0)
    return (sumx//count, sumy//count)

def find_slant(img):
    h,w = img.shape[:2]
    x1,y1 = center_of_mass(img[:,:w//2])
    x2,y2 = center_of_mass(img[:,w//2:])
    
    if(x2==x1):
        return 10000000000
    return (y2-y1)/(x2-x1)

# print("Mass Ratio -> ",find_mass(mask))
# print("closed area -> ",find_closed_area(mask))
# print("centroid ratio -> ",find_centroid_ratio(mask))
# print("baseline slant angle -> ",find_slant(mask))

# cv2.imshow('Thresholded Signature',mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



def get_features(img_name, name):
    img_path = "Signatures By Names/"+name+"/"+img_name+".jpg"
    print(img_path)
    img = cv2.imread(img_path)
    # cv2.imshow('photo',img)
    # cv2.waitKey(0)
    img = preprocess(img)
    true_features = [ find_aspect_ratio(img), find_mass(img) , find_closed_area(img) , find_centroid_ratio(img) , find_slant(img)]
    return true_features
