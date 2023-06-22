# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:32:00 2023

@author: pc07
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dbn.tensorflow import SupervisedDBNRegression
from sklearn.inspection import partial_dependence


def data_prerocessing (Input,Output):
    Input = Input /np.std(Input,axis = 0)
    (qt,Removal,finalPH,LeachingMg) = (Output[:,0],Output[:,1],Output[:,2],Output[:,3])
    
    fixpoint = dict(qt =  1.6675, 
                    Removal = np.std(Removal),
                    finalPH = np.std(finalPH),
                    LeachingMg = 2.56)
    
    qt_processed = np.emath.logn(1.6675,qt)
    Removal_processed = Removal/np.std(Removal)
    finalPH_processed = finalPH/np.std(finalPH)
    LeachingMg_processed = np.emath.logn(2.56,LeachingMg+1)    
    Output = np.stack((qt_processed,Removal_processed,finalPH_processed,LeachingMg_processed),axis = 1)
    
    return (Input,Output,fixpoint)

def cal_result(prediction,fix_point):
    
    qt_result = np.power(fix_point['qt'],prediction[:,0])
    Removal_result = prediction[:,1] * fix_point['Removal']
    finalPH_result = prediction[:,2] * fix_point['finalPH']
    LeachingMg_result = np.power(fix_point['LeachingMg'],prediction[:,3])-1
    result = np.stack((qt_result,Removal_result,finalPH_result,LeachingMg_result),axis = 1)
    
    return result

# Data inputing
data= pd.read_csv('F-removal-by-MgO-data-321data-points.csv')
data = pd.DataFrame.to_numpy(data)  

# Split arrays or matrices into random train and test subset 
X = data[:,0:15]
Y = data[:,15:19]

X_,Y_ ,fix_point= data_prerocessing(X,Y)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_, test_size=0.3, random_state=1)

# Training
regressor = SupervisedDBNRegression(hidden_layers_structure=[32,32],
                                    learning_rate_rbm=0.001,
                                    learning_rate=0.005,
                                    n_epochs_rbm=100,
                                    n_iter_backprop=40000,
                                    batch_size=512,
                                    activation_function='relu')

regressor = SupervisedDBNRegression.load('./Model/best_model_ver5.h5')


df = pd.read_csv('F-removal-by-MgO-data-321data-points.csv')
feat_name = df.columns.values
std = np.std(X,axis = 0)

# Iterate through each input feature
for i, feature_index in enumerate(range(15)):
    # Create the Partial Dependence plot
    display = partial_dependence(regressor, X_train, features=[feature_index])
    
    feat = np.matrix.transpose(display.get('values')[0]*std[i])
    target = cal_result(np.matrix.transpose(display.get('average')) , fix_point)
    
    save_data = {}  
    
    save_data[feat_name[i]] = feat
    save_data[feat_name[15]] = target[:,0]
    save_data[feat_name[16]] = target[:,1]
    save_data[feat_name[17]] = target[:,2]
    save_data[feat_name[18]] = target[:,3]
    
    data = pd.DataFrame(data = save_data)
    data.to_csv(feat_name[i]+ ' _ PartialDependencePlot.csv',index = False)