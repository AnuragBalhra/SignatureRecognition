from __future__ import division
# libraries to include 
# from matplotlib import pyplot as plt
from preprocess import *
from features import *
# from scipy import ndimage
import cv2
import numpy as np

from os import listdir
mypath = "Signatures By Names/"
print("searching for folder : ",mypath)
dirs = [f for f in listdir(mypath)]
#folders_cnt = len(dirs)
features = {}

'''
for folder in dirs:
    folder_path = mypath+folder+'/'
    print("Inside Folder : "+folder_path+"\n Processing Files : ")
    files = [f.split('.')[0] for f in listdir(folder_path)]
    temp = []
    for image in files:
        print("\t"+image + " .....")
        file_path = folder_path + image + ".jpg"
        img = cv2.imread(file_path)
        img = preprocess(img)
        print [ find_aspect_ratio(img), find_mass(img) , find_closed_area(img) , find_centroid_ratio(img) , find_slant(img)]
        temp.append( [ find_aspect_ratio(img), find_mass(img) , find_closed_area(img) , find_centroid_ratio(img) , find_slant(img)] )
    features[folder] = np.array(temp)

print(features)
'''



from sklearn import svm
from os import listdir
mypath = "Signatures By Names/"
# folders = [f for f in listdir(mypath)]

features = {}
actual_output = {}
# for folder in folders:
#     features[folder] = []
#     actual_output[folder] = []


input_file = open('train_images.txt','r')
for line in input_file:
    img_name, person_name, valid = line.split('\t')
    features[person_name] = []
    actual_output[person_name] = []

print "init features ",features


input_file = open('train_images.txt','r')

for line in input_file:
    img_name, person_name, valid = line.split('\t')
    img_feature = get_features(img_name, person_name)
    if(valid[0]=='1'):
        valid = 1
    else :
        valid = -1
    features[person_name].append(img_feature)
    actual_output[person_name].append(valid)

print "features:\n"
print features

svms = {}
for person,feature in features.items():
    if(len(feature)<=0):
        continue
    print person
    # clf = svm.SVC(kernel='rbf',C=5)
    clf = svm.SVC(kernel='linear')
    clf.fit(feature, actual_output[person])
    svms[person] = clf 
    import pickle
    # now you can save it to a file
    with open('svm_classifiers/'+person+'.pkl', 'wb') as f:
        pickle.dump(clf, f)

    predicted_output = clf.predict(feature)
    print "predicted output for trainig data : ",predicted_output
print svms
# exit()
# print "world"
final_avg = {x:np.mean(y,axis=0) for x,y in features.items()}
final_std = {x:np.std(y, axis=0) for x,y in features.items()}

print("final avg : ",final_avg)
print("final std : ",final_std)
# print(final_avg['Akshay Chaudhari'])
# np.savetxt('avg.txt', avg)
# np.savetxt('std.txt', std)
np.save('avg.npy',final_avg)
np.save('std.npy',final_std)
# np.save('svms.npy',svms)

print("saved the trained svm classifier for features in files svms.npy")
# print("saved the trained features in files avg.npy and std.npy")
