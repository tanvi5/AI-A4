#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 17:20:47 2018
@author: harsh
"""
import sys
import pickle
import numpy as np
import math
import pandas as pd
import random 
import itertools
from math import log
import operator

import time

rotations = [0,90,180,270]

'''
######################### Random Forest ##################################
'''

tree_count = 15
max_depth = 15  
current_depth = 0  


'''
This method reads images and converts them into an 8X8 matrix in which each cell in 8X8 has rgb values for a pixel
i.e. each cell has data arranged as (r,g,b)
[[(r1,g1,b1),(r2,g2,b2),.....(r8,g8,b8)]
[(r9,g9,b9),(r10,g10,b10),.....(r16,g16,b16)]
    .
    .
    .
[(r57,g57,b57),(r58,g58,b58),.....(r64,g64,b64)]    
'''     
def read_images(fname):
    
    file = open(fname, 'r')
    for line in file:
        
        matrix = []
        for i in range(8):
            matrix.append([0,0,0,0,0,0,0,0])
        values = line.split()
        for v in range(64):
            matrix[v//8][v%8] = [int(values[(3*v)+2]),int(values[(3*v)+3]),int(values[(3*v)+4])]
        images[int(values[1])].append([values[0],matrix])

'''
Here, we have calculated a feature as a subtraction of summation of nth and (8-n)th row/col, 
beleiving that the difference of pixel values will be a lot different for the top section of image and 
for bottom section of image. For example, outdoor pictures usually have lighter shades at top and darker shades at bottom.
We beleive, that maximum data relevant for guessing orientation will be present at the outer frame of picture.
So, new features as:
0: (sum(red pixel values of row 0) - sum(red pixel values of row 7))/25
1: (sum(red pixel values of col 0) - sum(red pixel values of col 7))/25
    .
    .
    .
    .
22: (sum(blue pixel values of row 3) - sum(blue pixel values of row 4))/25
23: (sum(blue pixel values of col 3) - sum(blue pixel values of col 4))/25
''' 
def calculate_features():
    global features
    for r in images.keys():
        for img,matrix in images[r]:
            
            for  num2 in range(4):
                    
                rhsum = 0
                rvsum = 0
                ghsum = 0
                gvsum = 0
                bhsum = 0
                bvsum = 0
                for num in range(8):
                
                    rhsum+=(matrix[num2][num][0]-matrix[(num2*(-1))-1][num][0])
                    rvsum+=(matrix[num][num2][0]-matrix[num][(num2*(-1))][0])
                    ghsum+=(matrix[num2][num][1]-matrix[(num2*(-1))][num][1])
                    gvsum+=(matrix[num][num2][1]-matrix[num][(num2*(-1))][1])
                    bhsum+=(matrix[num2][num][2]-matrix[(num2*(-1))][num][2])
                    bvsum+=(matrix[num][num2][2]-matrix[num][(num2*(-1))][2])
                features[6*num2].append(rhsum//25)
                features[(6*num2)+1].append(rvsum//25)
                features[(6*num2)+2].append(ghsum//25)
                features[(6*num2)+3].append(gvsum//25)
                features[(6*num2)+4].append(bhsum//25)
                features[(6*num2)+5].append(bvsum//25)
            features['rotation'].append(r)
            features['image'].append(img)
 
'''
Calculates entropy as sum(-p*log(p)).
'''    
def calculate_entropy(indices):
    entropy = 0
    orientation = {0:0,90:0,180:0,270:0}
    for i in indices:
        orientation[features['rotation'][i]] +=1
    for val in orientation.values():
        ratio = val/len(indices)
        if ratio!=0:
            entropy -= ratio*(log(ratio,2))   
    return entropy


class Node:
    def __init__(self, feature, val, true_branch=None, false_branch=None,decision = None):#question, true_branch, false_branch
        self.feature = feature              #Feature based on which split will take place
        self.val = val                      #Threshold value for feature 
        self.decision = decision            #Guessed orientation
        self.true_branch = true_branch      #Left branch
        self.false_branch = false_branch    #Right branch

'''
Returns guessed orientations for given input images based on majority
'''
def getdecision(indices):
    orientation = {0:0,90:0,180:0,270:0}
    for i in indices:
        orientation[features['rotation'][i]]+=1
    return max(orientation.items(), key=operator.itemgetter(1))[0]


################################################################################################
# We have referred 'https://www.youtube.com/watch?v=LDRbO9a6XPU' to write understand the working 
# of random forest. 
################################################################################################ 
        
'''
Partitions data into two subtrees based on value of feature
'''
def partition(feature, val,indices):
    true_rows, false_rows = [], []
    for index in indices:
        if features[feature][index] <= val:
            true_rows.append(index)
        else:
            false_rows.append(index)
    return true_rows, false_rows
    

def find_best_split(indices,parent_num):
    #global count
    #count+=1
    #print count
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain 
    best_question = None  # keep train of the feature / value that produced it
    current_entropy = calculate_entropy(indices)

    for feature in features.keys():  # for each feature
        if feature == 'rotation':
            continue
        #print features[feature]
        values = set([f for f in features[feature]])  # unique values in the column
        #print values
        for val in values:  # for each value

            # try splitting the dataset
            true_rows, false_rows = partition(feature, val,indices)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            
            
            info_gain = current_entropy - ((calculate_entropy(true_rows)*len(true_rows)/len(indices)) +
                                               (calculate_entropy(false_rows)*len(false_rows)/len(indices)))
            if info_gain >= best_gain:
                #print 'info gain update:',info_gain, (feature, val)
                #print 'false rows:',[features['rotation'][i] for i in false_rows  ]
                #print 'true rows:',[features['rotation'][i] for i in true_rows  ]
                best_gain, best_question = info_gain, (feature, val)
    return best_gain, best_question
    
    
def build_tree(indices,num_of_rows_in_parent,current_depth):
    current_depth+=1
    

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(indices,num_of_rows_in_parent)#[i for i in range(len(features['rotation']))]
    #print gain, question 
    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if max_depth<current_depth or gain == 0:
        #current_depth-=1
        return Node(None,None,decision = getdecision(indices))

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(question[0],question[1],indices)
    #print len(true_rows)
    #print len(false_rows)
    # Recursively build the true branch.
    true_branch = build_tree(true_rows,len(indices),current_depth)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows,len(indices),current_depth)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Node(question[0],question[1], true_branch, false_branch)    

################################################################################################
# Reference to https://www.youtube.com/watch?v=LDRbO9a6XPU Ends here
################################################################################################

'''
Returns decision for current tree after traversing 
'''
def traverse_tree(parent_node,index):
    current_node = parent_node
    while current_node.true_branch!=None or current_node.false_branch!=None:
        if features[current_node.feature][index] <= current_node.val:
            current_node = current_node.true_branch
        else:
            current_node = current_node.false_branch
    #print current_node.decision
    return current_node.decision

'''
Return majority response from all trees
'''  
def find_majority_vote(index,forest):
    votes = []
    for parent_node in forest:
        votes.append(traverse_tree(parent_node,index))      #get decision by traversing current tree
    return max(set(votes), key=votes.count)                 #return orientation which has max votes
        
            
            
def train_randomForest(trainFile, modelFile):
    global features
    global images
    features['rotation']=[]
    features['image']=[]
    for f in range(24):
        features[f] = []
    read_images(trainFile)
    calculate_features()
    total_images = int(len(features['rotation'])/4)         #Actual number of images, assuming that each image is present 4 times with different orientations
    sot = int(total_images/tree_count)#size_of_tree         #size_of_tree
    
    forest = []
    for i in range(tree_count):
        indices = [j for j in range(i*sot,(i+1)*sot)]+[j for j in range(total_images+(i*sot),total_images+(i+1)*sot)]+[j for j in range((2*total_images)+i*sot,(2*total_images)+(i+1)*sot)]+[j for j in range((3*total_images)+i*sot,(3*total_images)+(i+1)*sot)]
        forest.append(build_tree(indices,sot,0)) #find_best_split([i for i in range(len(features['rotation']))])
    
    model_f = open(modelFile, "wb")
    pickle.dump(forest, model_f)
    model_f.close()

def test_randomForest(testFile, modelFile):
    global features
    global images
    read_f = open(modelFile, "rb")
    forest = pickle.load(read_f)
    read_f.close()
    
    
    #features = {}
    features['rotation']=[]
    features['guessed']=[]
    features['image']=[]
    for f in range(24):
        features[f] = []
    
    read_images(testFile)
    calculate_features()   
    for row in range(len(features[0])):
        features['guessed'].append(find_majority_vote(row,forest))
        
    output_df = pd.DataFrame()
    output_df[0] = features['image']
    output_df[1] = features['guessed']
    output_df.to_csv("output.txt", sep = " ", header = None, index = None)
    print("Random Forest Accuracy : ", find_accuracy(features['rotation'],features['guessed']))

def find_accuracy(actual_result, new_result):
    correct = 0
    for i in range(0,len(actual_result)-1):
        if actual_result[i] == new_result[i]:
            correct += 1
    accuracy = (float(correct)/len(actual_result)) *100  #accuracy 
    return accuracy

'''
######################### KNN ##################################
'''

# For each test data, calculate distance with training data
# Consider top k values of training having lowest distance
# Take voting of k elements to determine class of test data
def knn(training_features, test_features, training_y, k):
    
    test_data = np.array(test_features)
    train_data= np.array(training_features)

    
    test_results = []
    for i in range(len(test_data)):
            rotations = {0:0,90:0,180:0,270:0}
            
            distances = np.linalg.norm(train_data- test_data[i], axis = 1)
            k_elements = sorted(distances)[:k]
            for j in k_elements:
                 index = np.argwhere(distances==j)
                 if(len(index)!=1):
                     for x in index:
                         rotations[training_y[x[0]]] +=1
                 rotations[training_y[index[0][0]]] +=1
            test_results.append(max(rotations, key = rotations.get))
    return test_results
 
'''
######################### Adaboost ##################################
'''
# FUnction that will randomly generate pixel values for generating decision stumps
def generate_hypothesis():
    hypothesis = []
    for i in range(400):
        pixel1 = random.randint(0,191)
        pixel2 = random.randint(0,191)
        while((pixel1,pixel2) in hypothesis):
            pixel1 = random.randint(0,191)
            pixel2 = random.randint(0,191)
        hypothesis.append((pixel1,pixel2))
    return hypothesis

# Generate data set function will generate combinations of classes to convert it into binary classification type problem
# list_combinations - stores list of combinations like 0-90, 0-180, 0-270, 90-180, 90-270 and 180-270
# combinations_training will generate training data specific to combination
# combined_df will store hypothesis specific to combination
def generate_datasets(trainingSet):
    # Group by class label
    df_grouped = trainingSet.groupby([1], axis = 0)
    
    combinations_dict = {}
    combinations_training = {}
    hypothesis = generate_hypothesis()
    
    # Generate all possible combinations of class labels
    # This will generate 4C2 combinations
    list_combinations = [(x,y) for x,y in itertools.combinations(rotations, 2)]
    
    # Assign hypothesis for each combination and respective training data
    for x,y in list_combinations:
        combinations_dict[(x,y)] = hypothesis
        combined_df = pd.concat([df_grouped.get_group(x), df_grouped.get_group(y)])
        combinations_training[(x,y)] = combined_df
    return list_combinations, combinations_dict, combinations_training

# This function will calculate error and correctly classified index based on decision stumps
def calculate_error(hypo, dataSet, weights, x, y):
    dataSet = np.array(dataSet)
    error = 0
    classified_indexes = []
    for index in range(len(dataSet)):
        # Positive if val1 >= val 2 and positive given for bigger number i.e. y
        # Hanced classified data if output is positive and class is positive OR output is negative and class is negative
        if((dataSet[index][2+hypo[0]] >= dataSet[index][2+hypo[1]] and dataSet[index][1] == y) or ((dataSet[index][2+hypo[0]] < dataSet[index][2+hypo[1]] and dataSet[index][1] == x))): 
            classified_indexes.append(index)
        else:
            # Update total error by adding respective weight of unclassified data
            error = error+ weights[index]
    return error, classified_indexes


# This is main train function that will calculate fincal decision stumps and associated weights
# It returns Dictionary that stores weighed decision stumps data fpr each combination
result_values = {}
def train_data(list_combinations, combinations_dict, combinations_training):
    for x,y in list_combinations:
        dataSet = combinations_training.get((x,y))
        weights = [1/len(dataSet) for i in range(len(dataSet))]
        weights = np.array(weights)
        output = []
        for  hypo in combinations_dict.get((x,y)):
            error, classified_indexes = calculate_error(hypo, dataSet,weights, x, y)
            if error<=0.5:
                for i in classified_indexes:
                    weights[i]=weights[i] * (error/(1-error))
                
                # normalize weights
                total = sum(weights)
                weights = weights / total
                
                # Find alpha value for hypothesis
                alpha = math.log((1-error)/error)
                
                output.append((hypo,alpha))
                
        # Dictionary that stores weighed decision stumps data fpr each combination
        result_values[(x,y)] = (output)
    return result_values





features = {}


#file_type = "test"
#file1 = "test-data.txt" # train-data.txt test-data.txt
#file2 = "best_model.txt" # nearest_model.txt, adaboost_model.txt, forest_model.txt,best_model.txt
#model = "best" # nearest, adaboost, forest, best

file_type = (sys.argv[1])
file1 = sys.argv[2]
file2 = sys.argv[3]
model = sys.argv[4]

fileSet = pd.read_csv(file1, header = None, sep=" ")
list_combinations = [(x,y) for x,y in itertools.combinations(rotations, 2)]
if(model == 'nearest' or model=='best'):
    k_val = 48
    if(file_type == 'train'):
        # Generate model file from training
        open("nearest_model.txt", "w").writelines([l for l in open(file1).readlines()])
        open("best_model.txt", "w").writelines([l for l in open(file1).readlines()])
        
        
    else:
        training_features = pd.read_csv(file2, header = None, sep=" ")
        # Read model file 
        training_y = training_features[1]
        training_features.drop([0,1], axis=1, inplace = True)
        training_features.columns = range(training_features.shape[1])
        
        # Read test file
        test_features = fileSet
        output_df = pd.DataFrame()
        output_df[0] = fileSet[0]
        test_y = test_features[1]
        test_features.drop([0,1], axis=1, inplace = True)
        test_features.columns = range(test_features.shape[1])
        
        # Find test results
        test_results = knn(training_features,test_features,training_y,k_val)

        output_df[1] = test_results
        
        output_df.to_csv("output.txt", sep = " ", header = None, index = None)
        # Find accuracy
        auc = find_accuracy(test_y, test_results)
        print("KNN accuracy - ", auc)
        
elif(model == 'adaboost'):
    if(file_type == 'train'):
        trainingSet = fileSet
        list_combinations, combinations_dict, combinations_training = generate_datasets(trainingSet)
        result_values = train_data(list_combinations, combinations_dict, combinations_training)
        model_f = open("adaboost_model.txt", "wb")
        pickle.dump(result_values, model_f)
        model_f.close()
        
#        model_f.write(json.dumps(result_values))
    else:
        testSet = fileSet
        model_f = open(file2, "rb")
        result_values = pickle.load(model_f)
        model_f.close()
        
        
        result = []
        dataSet = np.array(testSet)
        for index in range(len(dataSet)):
            pred = {0: 0, 90: 0, 180: 0, 270: 0}
            for x,y in list_combinations:
                sign_total = 0
                for hypo in list(result_values.get((x,y))):
                    # As per hypothesis if val1 >= val 2 it is positive and for each combination bigger number y is positive
                    
                    label = 1 if(dataSet[index][2+hypo[0][0]] >= dataSet[index][2+hypo[0][1]]) else -1
                    sign_total += hypo[1]*label
                # Find sign based on weighest decision stumps total value
                if(sign_total >= 0):
                    pred[y] += 1
                else:
                    pred[x] += 1
            # Take voting
            out = max(pred, key = pred.get)
            result.append(out)
            
        output_df = pd.DataFrame()
        output_df[0] = fileSet[0]
        output_df[1] = result
        output_df.to_csv("output.txt", sep = " ", header = None, index = None)    
        print("Adaboost accuracy - ", find_accuracy(testSet[1],result))
        
        
elif(model == 'forest'): 
    features = {}
    images = {0:[],90:[],180:[],270:[]}
    if(file_type == 'train'):
        train_randomForest(file1, file2)
    else:
        test_randomForest(file1, file2)
else:
    print("No such model")
