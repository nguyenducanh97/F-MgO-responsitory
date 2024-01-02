import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dbn.tensorflow import SupervisedDBNRegression


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

df= pd.read_csv('F-removal-by-MgO-data-321data-points.csv')
data = pd.DataFrame.to_numpy(df)  

#sSplit arrays or matrices into random train and test subset 
X = data[:,0:15]
Y = data[:,15:19]

X_,Y_ ,fix_point= data_prerocessing(X,Y)

kfold = RepeatedKFold(n_splits=3, n_repeats=3, random_state=1)
num_of_fold = 1
Metrics = np.zeros((10, 3))
for train, test in kfold.split(X_,Y_):
    print('FOLD',num_of_fold)
    
    X_train = X_[train]
    Y_train = X_[train]
    X_test = X_[test]
    Y_test = Y_[test]
    
    regressor = SupervisedDBNRegression(hidden_layers_structure=[32,16,8,4],
                                    learning_rate_rbm=0.001,
                                    learning_rate=0.003,
                                    n_epochs_rbm=100,
                                    n_iter_backprop= 1000,
                                    batch_size=512,
                                    activation_function='relu')
    
    regressor.fit(X_train, Y_train)
    #fit(regressor,data = (X_train, Y_train), verbose= false,repeat_task = 20, early_stop = true, save_model = true, link = '\model\save' )
    
    #evaluate model
    predictions = regressor.predict(X_test)  
    predictions = cal_result(predictions,fix_point)
    Y_t = cal_result(Y_test,fix_point)
    
    y_test_f = np.ndarray.flatten(Y_t)
    predictions_f = np.ndarray.flatten(predictions)
    
    # Metrics
    mae = mean_absolute_error(y_test_f, predictions_f)
    rmse = np.sqrt(mean_squared_error(y_test_f, predictions_f))
    R_score = r2_score(y_test_f, predictions_f)
    
    Metrics[num_of_fold-1] = np.array([[mae,rmse,R_score]])
    print('Metrics:','\t MAE:', mae,'\t RMSE: ', rmse, '\t R_score: ', R_score)
    
    save = np.concatenate((predictions,Y_t),axis=1)
    np.savetxt('result of testing fold'+str(num_of_fold)+'.csv', save, 
           header = "qt_predict, Removal_predict, finalPH_predict, LeachingMg_predict, qt, Removal, finalPH, LeachingMg", 
           delimiter=",",comments='' )
    regressor.save('model_fold'+str(num_of_fold)+'.h5')
    num_of_fold = num_of_fold+1