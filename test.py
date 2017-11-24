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
    # print(img_path)

    test_features = get_features(img_name, name)
    average = true_avg[name]
    sd = true_std[name]
    count = 0
    num_features = len(test_features)
    print "calculated features : \n",test_features
    for i in range(num_features):
        upper_bound = average[i]+sd[i]
        lower_bound = average[i]-sd[i]
        # print(lower_bound,' ',test_features[i],' ', upper_bound)
        if(test_features[i]<=upper_bound and test_features[i]>=lower_bound):
            count = count + 1
    # print("count=>",count)
    if(count>=num_features//2):
        return "Genuine"
    else :
        return "Forged"




from os import listdir
mypath = "test_images/"
input_file = open('test_images.txt','r')
actual_output = {}
predicted_output = {}

print "testing begins .....\n"

for line in input_file:
    if(line[0]=='\n' or line[0]=='#'):
        continue
    img_name, person_name, valid = line.split('\t')
    actual_output[person_name] = []
    predicted_output[person_name] = []

input_file = open('test_images.txt','r')

for line in input_file:
    if(line[0]=='\n' or line[0]=='#'):
        continue
    img_name, person_name, valid = line.split('\t')
    # print("valid",valid)
    predicted = test(img_name, person_name)
    if(valid[0]=='1'):
        valid = "Genuine"
    else :
        valid = "Forged"
    actual_output[person_name].append(valid)
    predicted_output[person_name].append(predicted)
    print "expected output : ",valid,'\n'
print "expected output : ",actual_output
print "predicted output : ",predicted_output
print ""
#images = [f for f in listdir(mypath)]


for person in actual_output:
    count = 0
    total = len(actual_output[person])
    for i in range(total):
        # print "adsfaagjdflvasjdvsdjcndo ",actual_output[person][i], predicted_output[person][i]
        if(actual_output[person][i] == predicted_output[person][i]):
            count = count + 1
    if(total>0):
        print "using Statistical analysis : "
        print "predicted : ",count , " signatures accurately, out of total ",total," tested signatures" 
        print "Accuracy : ",(count/total)*100," % for signatures of ", person ,'\n'


'''

Code for using SVM Classifiers for predicting accurate signatures

'''

from sklearn import svm
from os import listdir
mypath = "Signatures By Names/"

'''

Initialize variables for SVM Testing

'''
features = {}
actual_output = {}

input_file = open('test_images.txt','r')

for line in input_file:
    if(line[0]=='\n' or line[0]=='#'):
        continue
    img_name, person_name, valid = line.split('\t')
    features[person_name] = []
    actual_output[person_name] = []

# print "init features:\n"
# print features


input_file = open('test_images.txt','r')

for line in input_file:
    if(line[0]=='\n' or line[0]=='#'):
        continue
    img_name, person_name, valid = line.split('\t')
    img_feature = get_features(img_name, person_name)
    print "calculated features : \n",img_feature
    print "expected output : ",valid,'\n'

    # predicted = test_with_svm(img_name, person_name)
    if(valid[0]=='1'):
        valid = "Genuine"
    else :
        valid = "Forged"
    features[person_name].append(img_feature)
    actual_output[person_name].append(valid)

# print "features:\n"
# print features
print "expected output : ",actual_output,'\n'

for person,feature in features.items():
    if(len(feature)<=0):
        continue
    print "recognising signatures for : ",person
    import pickle
    with open('svm_classifiers/'+person+'.pkl', 'rb') as f:
        clf = pickle.load(f)
    predicted_output = clf.predict(feature)

    predicted_output = ["Genuine" if(x==1) else "Forged" for x in predicted_output]
    print "predicted output for '",person,"' are : ",predicted_output," using svm classifier "

    correct = 0
    total = len(actual_output[person])
    for x,y in zip(actual_output[person], predicted_output):
        if(x==y):
            correct = correct + 1
    if(total>0):
        # print "using SVM for person : ",person
        print "predicted : ",correct , " correct signatures out of total ",total," tested files"
        # print "Accuracy : ",(correct/total)*100," %"
    print "Total Accuracy for '",person,"' using SVM Classifiers : ",(correct/total)*100," %\n"



