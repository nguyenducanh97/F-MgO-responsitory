import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dbn.tensorflow import SupervisedDBNRegression
from sklearn.inspection import permutation_importance



# DATA PREPROCESSING

# Data preprocessing by normalization and grey level transformation
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

# Testing and training data set before being preprocessing



# TRAINING MODELS

# Training
regressor = SupervisedDBNRegression(hidden_layers_structure=[32,32],
                                    learning_rate_rbm=0.001,
                                    learning_rate=0.005,
                                    n_epochs_rbm=100,
                                    n_iter_backprop=40000,
                                    batch_size=512,
                                    activation_function='relu')
regressor.fit(X_train, Y_train)
regressor._fine_tuning(X_train, Y_train)

# Save the model
regressor.save('./DBN-model/best_model_ver5.h5')

#regressor = SupervisedDBNRegression.load('./DBN-model/best_model_ver1.h5')



# EVALUATION PREDICTION PERFORMANCE

# Evaluate model
predictions = regressor.predict(X_test)
predictions = cal_result(predictions,fix_point)
Y_t = cal_result(Y_test,fix_point)
y_test_f = np.ndarray.flatten(Y_t)
predictions_f = np.ndarray.flatten(predictions)

# Evaluation metrics
mae = mean_absolute_error(y_test_f, predictions_f)
rmse = np.sqrt(mean_squared_error(y_test_f, predictions_f))
R_score = r2_score(y_test_f, predictions_f)
print(mae,rmse,R_score)

# Plot evaluation diagrams
fig = plt.figure()
plt.figure(figsize = (40, 20))
for index in range(4):
    plt.subplot(2,2,index+1)
    plt.plot(predictions[:,index],label = 'Predict')
    plt.plot(Y_t[:,index],label = 'Actual')
    plt.legend(['Predict', 'Real'])

# Save preprocessed data to CSV
preprocessed_data = np.concatenate((X_, Y_), axis=1)
df_preprocessed_data = pd.DataFrame(preprocessed_data)
df_preprocessed_data.to_csv('Treated-set-F-removal-by-MgO-data.csv', index=False)
print('Successfully saved Treated-set-F-removal-by-MgO-data.csv:', preprocessed_data.shape)



# FEATURE IMPORTANCE

# Compute feature importances using permutation importance
result = permutation_importance(regressor, X_test, Y_test, n_repeats=10, random_state=0)
importances = result.importances_mean
feature_importances = pd.DataFrame(data={'feature': range(X_train.shape[1]), 'importance': importances})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
feature_importances['importance'] = feature_importances['importance'] / feature_importances['importance'].sum()

# Save the DataFrame to a CSV file
feature_importances.to_csv('FeIm-data-DBN-ver1.csv', index=False)
print('Successfulluy saved FeIm-data-DBN-ver1.csv')



#SHAP BEESWARM PLOT FOR SINGLE OUTPUT FEATURE

import shap
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'
explainer = shap.KernelExplainer(regressor.predict, X_)
shap_values = explainer.shap_values(X_)
column_names = list(pd.read_csv('F-removal-by-MgO-data-321data-points.csv').columns[:15])

# Create beeswarm plots for each output feature with labeled features
for i in range(Y_.shape[1]): 
    plt.figure(figsize=(16, 9), dpi=100)  
    shap.summary_plot(shap_values[i], features=X_, feature_names=column_names, plot_type='violin')
    plt.title(f"SHAP Beeswarm Plot for Output Feature {i+1}", fontdict={'fontname': 'Arial', 'fontsize': 18})
    plt.xlabel("SHAP Value", fontdict={'fontname': 'Arial', 'fontsize': 14}) 
    plt.ylabel("Feature", fontdict={'fontname': 'Arial', 'fontsize': 14})  
    plt.xticks(fontname='Arial', fontsize=12)  
    plt.yticks(fontname='Arial', fontsize=12) 
    plt.tight_layout()
    plt.show()
