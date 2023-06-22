# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:21:03 2023

@author: pc07
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import seaborn as sns
from scipy import stats
import joblib
from sklearn.inspection import permutation_importance
import sklearn
from tensorflow import keras



# DATA PREPROCESSING

# Data preprocessing by normalization and grey level transformation
def data_prerocessing (Input,Output):
    Input = Input/np.std(Input,axis = 0)
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

# Loading data
df= pd.read_csv('F-removal-by-MgO-data-321data-points.csv')
data= pd.read_csv('F-removal-by-MgO-data-321data-points.csv')
data = pd.DataFrame.to_numpy(data)
print(data.shape)

# Data preprocessing
X = data[:,0:15]
Y = data[:,15:19]
X_,Y_ ,fix_point= data_prerocessing(X,Y)

# Split arrays or matrices into random train and test subset (training 70%, testing 30%)
X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.3, random_state=1)
X_train_, X_test_, y_train_, y_test_ = train_test_split(X, Y, test_size=0.3, random_state=1)



# DATA VISUALIZATION

# calculate the majority value for each column and the number of data points that are different from the majority value in each column
majority = df.mode().iloc[0]
diff_counts = (df != majority).sum()
print('Successfully calculated number of data different from the rest')

fig = plt.figure()
plt.figure(figsize = (20, 20))
for i in range(19):
    plt.subplot(4,5,i+1)
    plt.plot(data[:,i])
    
# Calculate correlation coefficient
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .6), xycoords=ax.transAxes,
               size = 30)

sns.set_context(font_scale=3)

# Pair grid set up
g = sns.PairGrid(df)
g.map_upper(plt.scatter, s=12, color = 'blue')
g.map_diag(plt.hist, color = 'blue')
g.map_lower(corrfunc);
for ax in g.axes.flat:
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
plt.savefig('Correlation Matrix Plot with R-score.png', dpi=500, bbox_inches='tight')



# DNN WITH MULTIPLE REGRESSIONS

# DNN version 1
ANN_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=8, activation='relu'),
    tf.keras.layers.Dense(units=4, activation='relu'),
])

# DNN version 2
Input = tf.keras.Input(shape=(None,15))
X = tf.keras.layers.Dense(units=32, activation='sigmoid')(Input)
X = tf.keras.layers.Dense(units=16, activation='relu')(X)
X1 = tf.keras.layers.Dense(units=2, activation='relu')(X)
X1 = tf.keras.layers.Dense(units=1, activation='relu')(X1)

X2 = tf.keras.layers.Dense(units=2, activation='relu')(X)
X2 = tf.keras.layers.Dense(units=1, activation='relu')(X2)

X3 = tf.keras.layers.Dense(units=2, activation='relu')(X)
X3 = tf.keras.layers.Dense(units=1, activation='relu')(X3)

X4 = tf.keras.layers.Dense(units=2, activation='relu')(X)
X4 = tf.keras.layers.Dense(units=1, activation='relu')(X4)
Output = tf.keras.layers.Concatenate()([X1,X2,X3,X4])

ANN_model = tf.keras.Model(Input,Output)



# TRAINING MODELS, EVALUATION

# Define opimizaion method
opt = tf.keras.optimizers.Adam(learning_rate=0.005)
# Define loss function
ANN_model.compile(optimizer=opt, loss=losses.MeanSquaredError())
# Training models
ANN_model.fit(X_train, y_train, validation_data = (X_test,y_test), batch_size=512, epochs=4000, shuffle=True)

# Save models
ANN_model.save('./DNN-ML-save-model/save_model/saved_model_DNN_32_16_4x2_4x1')
#ANN_model.save('./DNN-ML-save-model/save_model/saved_model_DNN_32_16_8_4')

# Load model
model = joblib.load('./DNN-ML-save-model/ElasticNet Regression.pkl')
#model = joblib.load('./DNN-ML-save-model/Extra Trees.pkl')
#model = joblib.load('./DNN-ML-save-model/Random Forest.pkl')
#model = joblib.load('./DNN-ML-save-model/Lasso.pkl')
#model = joblib.load('./DNN-ML-save-model/BaggingRegressor.pkl')
#model = joblib.load('./DNN-ML-save-model/KNeighborsRegressor.pkl')
#ANN_model = keras.models.load_model('./DNN-ML-save-model/save_model/saved_model_DNN_32_16_4x2_4x1')
#ANN_model = keras.models.load_model('./DNN-ML-save-model/save_model/saved_model_DNN_32_16_8_4')

#Change model
predictions = model.predict(X_test)
predictions = cal_result(predictions,fix_point)
Y_t = cal_result(y_test,fix_point)
y_test_f = np.ndarray.flatten(Y_t)
predictions_f = np.ndarray.flatten(predictions)

# Metrics
mae = mean_absolute_error(y_test_f, predictions_f)
rmse = np.sqrt(mean_squared_error(y_test_f, predictions_f))
R_score = r2_score(y_test_f, predictions_f)
print([mae,rmse,R_score])



# FEATURE IMPORTANCE

def keras_scoring(model, X, y):
    y_pred = model.predict(X)
    return -mean_squared_error(y, y_pred)

# Compute feature importance (change model)
result = permutation_importance(model, X_test, y_test, scoring=keras_scoring, n_repeats=10, random_state=42)
importances = result.importances_mean
feature_importances = pd.DataFrame(data={'feature': range(X_train.shape[1]), 'importance': importances})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
feature_importances['importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
feature_importances.to_csv('FeIm-DNN-ver1.csv', index=False)
print('Successfulluy saved FeIm-DNN-ver1.csv')



# TRADITIONAL MACHINE LEARNING MODELS

# Evaluate Classical ml models by training on training set and testing on testing set
def evaluate_classical_ML(X_train, X_test, y_train, y_test):
    # Names of models
    model_name_list = ['Multiple Linear Regression', 'ElasticNet Regression',
                     'Random Forest', 'Extra Trees', 'Lasso', 'Ridge',
                       'BaggingRegressor', 'KNeighborsRegressor']

# Instantiate the models
    model1 = LinearRegression()
    model2 = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model3 = RandomForestRegressor(n_estimators=10)
    model4 = ExtraTreesRegressor(n_estimators=20)
    model5 = Lasso(alpha=0.5)
    model6 = Ridge(alpha=0.5)
    model7 = sklearn.ensemble.BaggingRegressor(n_estimators=50)
    model8 = sklearn.neighbors.KNeighborsRegressor()


# Dataframe for results
    results = pd.DataFrame(columns=['mae', 'rmse', 'R-Squared'], index = model_name_list)
    plt.figure(figsize = (16, 16))
    plt.suptitle('Classical ML Predictions')

# Train and predict with each model
    for i, model in enumerate([model1, model2, model3, model4,model5,model6, model7, model8]):
        model.fit(X_train, y_train)

        joblib.dump(model,'/content/drive/MyDrive/Colab Notebooks/save model'+ model_name_list[i]+'.pkl')

        predictions = model.predict(X_test)
        
#Calculate result
        predictions = cal_result(predictions,fix_point)
        Y_t = cal_result(y_test,fix_point)
        y_test_f = np.reshape(Y_t,(384,1))
        predictions_f = np.reshape(predictions,(384,1))

# Metrics
        mae = mean_absolute_error(y_test_f, predictions_f)
        rmse = np.sqrt(mean_squared_error(y_test_f, predictions_f))
        R_score = r2_score(y_test_f, predictions_f)

# Insert results into the dataframe
        model_name = model_name_list[i]
        results.loc[model_name, :] = [mae, rmse, R_score]
        plt.subplot(4,3,i+1)
        plt.plot(y_test_f, color = 'red', label = 'Real data')
        plt.plot(predictions_f)
        plt.legend(['Actual', model_name_list[i]], fontsize = 10)

    plt.savefig('Classical ML Predictions.png', dpi=500, bbox_inches='tight')
    plt.show()

    return results

    evaluate_classical_ML(X_train, X_test, y_train, y_test)



# EVALUATION OF TRAINING/TESTING DATA SET

# Load the trained model
#model = joblib.load('./DNN-ML-save-model/ElasticNet Regression.pkl')
#model = joblib.load('./DNN-ML-save-model/Extra Trees.pkl')
#model = joblib.load('./DNN-ML-save-model/Random Forest.pkl')
#model = joblib.load('./DNN-ML-save-model/Lasso.pkl')
#model = joblib.load('./DNN-ML-save-model/BaggingRegressor.pkl')
#model = joblib.load('./DNN-ML-save-model/KNeighborsRegressor.pkl')
ANN_model = keras.models.load_model('./DNN-ML-save-model/save_model/saved_model_DNN_32_16_4x2_4x1')
#ANN_model = keras.models.load_model('./DNN-ML-save-model/save_model/saved_model_DNN_32_16_8_4')

def get_predictions(ANN_model, X, y):
    y_true = y_test_ if y is y_test_ else y_train_
    y_pred = ANN_model.predict(X)
    return y_pred, y_true

#y_pred, y_true = get_predictions(ANN_model, X_test, y_test_)
y_pred, y_true = get_predictions(ANN_model, X_train, y_train_)

# Data processing
y_pred = cal_result(y_pred,fix_point)
output_data = {'qt': y_pred[:,0] , 'R': y_pred[:,1], 'finalpH' : y_pred[:,2], 'leachingMg': y_pred[:,3]}
output_data = pd.DataFrame(data=output_data)
output_path = './DNN-ML-save-model/save_model/predict_test.csv'
output_data.to_csv(output_path, index=False)

for index in range(4):
    y_pred_f = y_pred[:,index]
    y_true_f = y_true[:,index]
    mae = mean_absolute_error(y_pred_f, y_true_f)
    rmse = mean_squared_error(y_true_f, y_pred_f, squared=False)
    r2 = r2_score(y_pred_f, y_true_f)

# Print the evaluation metrics
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print("========================")

