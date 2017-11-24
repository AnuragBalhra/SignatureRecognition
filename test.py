from __future__ import division
from matplotlib import pyplot as plt
from preprocess import *
from features import *
# from scipy import ndimage
import cv2
import numpy as np

from os import listdir

true_avg = np.load('avg.npy').item()
true_std = np.load('std.npy').item()
# svms = np.load('svms.npy').item()

# print(load)

def test(img_name, name):
    img_path = "Signatures By Names/"+name+"/"+img_name+".jpg"
    print(img_path)

    test_features = get_features(img_name, name)
    average = true_avg[name]
    sd = true_std[name]
    count = 0
    for i in range(len(test_features)):
        upper_bound = average[i]+sd[i]
        lower_bound = average[i]-sd[i]
        print(lower_bound,' ',test_features[i],' ', upper_bound)
        if(test_features[i]<=upper_bound and test_features[i]>=lower_bound):
            count = count + 1
    print("count=>",count)
    if(count>=3):
        return True
    else :
        return False

def test_with_svm(img_name, name):
    img_path = "Signatures By Names/"+name+"/"+img_name+".jpg"
    print(img_path)

    test_features = get_features(img_name, name)
    
    count = 0
    for i in range(len(test_features)):
        upper_bound = average[i]+sd[i]
        lower_bound = average[i]-sd[i]
        print(lower_bound,' ',test_features[i],' ', upper_bound)
        if(test_features[i]<=upper_bound and test_features[i]>=lower_bound):
            count = count + 1
    print("count=>",count)
    if(count>=3):
        return True
    else :
        return False






from os import listdir
mypath = "test_images/"
input_file = open('test_images.txt','r')
actual_output = []
predicted_output = []
for line in input_file:
    img_name, person_name, valid = line.split('\t')
    # print("valid",valid)
    predicted = test(img_name, person_name)
    if(valid[0]=='1'):
        valid = True
    else :
        valid = False
    actual_output.append(valid)
    predicted_output.append(predicted)
    print(predicted)
print "expected output : ",actual_output
print "predicted output : ",predicted_output
#images = [f for f in listdir(mypath)]


correct = 0
total = len(actual_output)

for x,y in zip(actual_output, predicted_output):
    if(x==y):
        correct = correct + 1
if(total>0):
    print "using Statistical analysis : \n"
    print("predicted : ",correct , " correct signatures out of total ",total," tested files")
    print("Accuracy : ",(correct/total)*100," %")
#images = [f for f in listdir(mypath)]



'''
Code for using SVM Classifiers for predicting accurate signatures
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

input_file = open('test_images.txt','r')

for line in input_file:
    img_name, person_name, valid = line.split('\t')
    features[person_name] = []
    actual_output[person_name] = []

print "init features:\n"
print features

input_file = open('test_images.txt','r')

for line in input_file:
    img_name, person_name, valid = line.split('\t')
    img_feature = get_features(img_name, person_name)
    print "got features : ",img_feature

    # predicted = test_with_svm(img_name, person_name)
    if(valid[0]=='1'):
        valid = True
    else :
        valid = False
    features[person_name].append(img_feature)
    actual_output[person_name].append(valid)

print "features:\n"
print features
print actual_output

for person,feature in features.items():
    if(len(feature)<=0):
        continue
    print "recognising signatures for : ",person
    import pickle
    with open('svm_classifiers/'+person+'.pkl', 'rb') as f:
        clf = pickle.load(f)
    predicted_output = clf.predict(feature)

    predicted_output = [True if(x==1) else False for x in predicted_output]
    print "predicted output for : ",person," are : ",predicted_output," using svm classifier  : \n",clf

    correct = 0
    total = len(actual_output[person])
    for x,y in zip(actual_output[person], predicted_output):
        if(x==y):
            correct = correct + 1
        if(total>0):
            # print "using SVM for person : ",person
            print "predicted : ",correct , " correct signatures out of total ",total," tested files"
            # print "Accuracy : ",(correct/total)*100," %"
    print "Total Accuracy using SVM Classifiers : ",(correct/total)*100," %"

exit()
print "world"
final_avg = {x:np.mean(y,axis=0) for x,y in features.items()}
final_std = {x:np.std(y, axis=0) for x,y in features.items()}
